#######################################################################
# Compute the score of generated samples
#######################################################################

import chz
import json
from tqdm import tqdm

import datasets
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig


@chz.chz
class CLIConfig:
    # dataset
    dataset_path: str = "test.json"


def compute_score(
    model_output: str, ground_truth: str, timeout_score: float = 0
) -> bool:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except Exception:
        pass
    except TimeoutException:
        ret_score = timeout_score

    return ret_score


def main(cli_config):
    # Load dataset
    ds = datasets.load_dataset(
        "json", data_files=cli_config.dataset_path, split="train"
    )

    # Compute scores
    new_samples = []
    for sample in tqdm(ds, desc="Computing scores", total=len(ds)):
        scores = []
        all_responses = sample["responses"]
        ground_truth = sample["answer"]
        for response in all_responses:
            score = compute_score(response, ground_truth)
            scores.append(score)

        new_samples.append(
            {
                "problem": sample["problem"],
                "answer": sample["answer"],
                "responses": sample["responses"],
                "scores": scores,
                "pass@k=0": not any(scores),
            }
        )

    # Save
    save_file = cli_config.dataset_path.replace(".json", "_scored.json")
    with open(save_file, "w", encoding="utf8") as f:
        for i in range(len(new_samples)):
            json.dump(new_samples[i], f, ensure_ascii=False)
            f.write("\n")

    print(f"Saved samples to {save_file}")


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    main(cli_config)
