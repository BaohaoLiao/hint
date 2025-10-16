#######################################################################
# Sampling generations from a dataset
#######################################################################

import os
import chz
import json
import numpy as np

import torch
import datasets
from vllm import LLM, SamplingParams


@chz.chz
class CLIConfig:
    # dataset
    dataset_name: str = "open-r1/OpenR1-Math-220k"
    split: str = "train"
    question_key: str = "problem"
    world_size: int = 1
    local_idx: int = 0

    # model
    model_name_or_path: str = "Qwen/Qwen2.5-Math-1.5B"
    max_model_length: int = 4096

    # sampling
    n: int = 16
    temperature: float = 1.0
    max_new_tokens: int = 4096
    seed: int = 42

    # Save
    output_path: str = "gen_dataset"


def main(cli_config):
    # Set seed
    torch.manual_seed(cli_config.seed)
    np.random.seed(cli_config.seed)

    # Load dataset
    if os.path.exists(cli_config.dataset_name) and os.path.isdir(
        cli_config.dataset_name
    ):
        ds = datasets.load_from_disk(cli_config.dataset_name)
    else:
        ds = datasets.load_dataset(cli_config.dataset_name, split=cli_config.split)
    
    # Process dataset
    ## Split the dataset equally among GPUs
    k, m = divmod(len(ds), cli_config.world_size)
    start = cli_config.local_idx * k + min(cli_config.local_idx, m)
    end = (cli_config.local_idx + 1) * k + min(cli_config.local_idx + 1, m)
    ds = ds.select(np.arange(start, end))
    print(f"Loaded {len(ds)} ([{start}, {end}]) samples from {cli_config.dataset_name}.")

    def make_prompt(example):
        system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
        question_suffix = " Let's think step by step and output the final answer within \\boxed{}."
        
        question = example[cli_config.question_key]
        question = question + question_suffix
        prompt_messages = [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": question}
        ]
        return {"prompt": tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)}

    ds = ds.map(make_prompt, num_proc=4)

    # Load model
    llm = LLM(
        model=cli_config.model_name_or_path,
        tokenizer=cli_config.model_name_or_path,
        dtype="bfloat16",
        max_model_len=cli_config.max_model_length,
        load_format="auto",
        seed=cli_config.seed,
    )
    tokenizer = llm.get_tokenizer()

    # Sampling
    sampling_params = SamplingParams(
        temperature=cli_config.temperature,
        top_p=1.0,
        max_tokens=cli_config.max_new_tokens,
        n=cli_config.n,
        stop_token_ids=[tokenizer.eos_token_id]
    )
    prompts = ds["prompt"]
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)

    # Save generations
    samples = []
    for i, output in enumerate(outputs):
        samples.append(
            {
                "problem": ds[i][cli_config.question_key],
                "answer": ds[i].get("answer"),
                "responses": [out.text for out in output.outputs],
            }
        )

    print(f"Collected {len(samples)} samples.")
    save_file = os.pah.join(cli_config.output_dir + str(cli_config.local_idx) + ".json")
    with open(save_file, "w", encoding="utf8") as f:
        for i in range(len(samples)):
            json.dump(samples[i], f, ensure_ascii=False)
            f.write("\n")

    print(f"Saved samples to {save_file}")


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    main(cli_config)
