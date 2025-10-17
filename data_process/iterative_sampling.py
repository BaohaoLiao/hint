#######################################################################
# Iterative sampling: Only resample problems that haven't passed yet
#######################################################################

import os
import chz
import json
import numpy as np

import torch
import datasets
from vllm import LLM, SamplingParams

from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig


@chz.chz
class CLIConfig:
    # dataset
    dataset_path: str = "validated.json"
    world_size: int = 1
    local_idx: int = 0

    # model
    model_name_or_path: str = "Qwen/Qwen2.5-Math-1.5B"
    max_model_length: int = 4096

    # sampling
    n: int = 8
    temperature: float = 1.0
    max_new_tokens: int = 4096
    seed: int = 42

    # iteration
    iteration: int = 1  # Current iteration number

    # Save
    output_dir: str = "gen_dataset"
    merged_file: str = "merged_all_iterations.json"


def compute_score(
    model_output: str, ground_truth: str, timeout_score: float = 0
) -> float:
    """Compute if model output matches ground truth."""
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"

    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except Exception:
        pass
    except TimeoutException:
        ret_score = timeout_score

    return ret_score


def load_previous_results(merged_file_path):
    """Load previous iteration results if they exist."""
    if not os.path.exists(merged_file_path):
        return {}

    results = {}
    with open(merged_file_path, "r", encoding="utf8") as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                # Use problem as key (assuming problems are unique)
                results[sample["problem"]] = sample

    return results


def get_failed_problems(previous_results):
    """Extract problems that haven't passed yet."""
    failed = []
    for problem, data in previous_results.items():
        if data.get("pass@k=0", True):  # If pass@k=0 is True, means all failed
            failed.append({"problem": problem, "answer": data["answer"]})
    return failed


def main(cli_config):
    # Sanitize output path
    if not os.path.exists(cli_config.output_dir):
        os.makedirs(cli_config.output_dir, exist_ok=True)

    # Set seed
    torch.manual_seed(cli_config.seed + cli_config.iteration)
    np.random.seed(cli_config.seed + cli_config.iteration)

    merged_file_path = os.path.join(cli_config.output_dir, cli_config.merged_file)

    # Load previous results
    print(
        f"[GPU {cli_config.local_idx}] Loading previous results from {merged_file_path}..."
    )
    previous_results = load_previous_results(merged_file_path)

    # Determine which problems to sample
    if cli_config.iteration == 1:
        # First iteration: load from original dataset
        ds = datasets.load_dataset(
            "json", data_files=cli_config.dataset_path, split="train"
        )
        print(
            f"[GPU {cli_config.local_idx}] First iteration: loaded {len(ds)} problems"
        )
    else:
        # Subsequent iterations: only sample failed problems
        failed_problems = get_failed_problems(previous_results)
        if not failed_problems:
            print(
                f"[GPU {cli_config.local_idx}] All problems have passed! No need for more iterations."
            )
            return

        ds = datasets.Dataset.from_list(failed_problems)
        print(
            f"[GPU {cli_config.local_idx}] Iteration {cli_config.iteration}: resampling {len(ds)} failed problems"
        )

    # Split dataset across GPUs
    k, m = divmod(len(ds), cli_config.world_size)
    start = cli_config.local_idx * k + min(cli_config.local_idx, m)
    end = (cli_config.local_idx + 1) * k + min(cli_config.local_idx + 1, m)
    ds = ds.select(np.arange(start, end))
    print(f"[GPU {cli_config.local_idx}] Processing samples [{start}, {end})")

    # Load model
    llm = LLM(
        model=cli_config.model_name_or_path,
        tokenizer=cli_config.model_name_or_path,
        dtype="bfloat16",
        max_model_len=cli_config.max_model_length,
        load_format="auto",
        seed=cli_config.seed + cli_config.iteration,
    )
    tokenizer = llm.get_tokenizer()

    def make_prompt(example):
        system_prompt = (
            "Please reason step by step, and put your final answer within \\boxed{}."
        )
        question_suffix = (
            " Let's think step by step and output the final answer within \\boxed{}."
        )
        question = example["problem"] + question_suffix
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        return {
            "prompt": tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
        }

    # Sampling
    sampling_params = SamplingParams(
        temperature=cli_config.temperature,
        top_p=1.0,
        max_tokens=cli_config.max_new_tokens,
        n=cli_config.n,
        stop_token_ids=[tokenizer.eos_token_id],
        seed=cli_config.seed + cli_config.iteration,
    )

    ds = ds.map(make_prompt, num_proc=4)
    prompts = ds["prompt"]
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)

    # Process outputs and compute scores
    samples = []
    for i, output in enumerate(outputs):
        problem = ds[i]["problem"]
        answer = ds[i]["answer"]
        new_responses = [out.text for out in output.outputs]

        # Compute scores for new responses
        new_scores = []
        for response in new_responses:
            score = compute_score(response, answer)
            new_scores.append(score)

        # Store only the new data for this iteration
        samples.append(
            {
                "problem": problem,
                "answer": answer,
                "responses": new_responses,
                "scores": new_scores,
                "iteration": cli_config.iteration,
            }
        )

    # Save current GPU's results for this iteration
    save_file = os.path.join(
        cli_config.output_dir, f"iter{cli_config.iteration}_{cli_config.local_idx}.json"
    )
    with open(save_file, "w", encoding="utf8") as f:
        for sample in samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write("\n")

    print(f"[GPU {cli_config.local_idx}] Saved {len(samples)} samples to {save_file}")

    # Properly cleanup the LLM to avoid NCCL warnings
    try:
        import gc
        import torch.distributed as dist
        
        # Destroy the LLM instance
        del llm
        
        # Cleanup distributed process group if it exists
        if dist.is_initialized():
            dist.destroy_process_group()
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"[GPU {cli_config.local_idx}] Cleaned up resources")
    except Exception as e:
        print(f"[GPU {cli_config.local_idx}] Warning during cleanup: {e}")


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    main(cli_config)
