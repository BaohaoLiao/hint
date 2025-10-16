#######################################################################
# Validate whether a question has a verifiable answer with math_verify.
# Filter out samples that cannot be verified.
#######################################################################

import os
import chz
import json
from tqdm import tqdm

import datasets
from math_verify.metric import math_metric
from math_verify.errors import TimeoutException
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig


@chz.chz
class CLIConfig:
    # dataset
    dataset_name: str = "open-r1/OpenR1-Math-220k"
    split: str = "train"
    answer_key: str = "answer"
    solution_key: str = "generations"

    # separator
    separator: str = "</think>"

    # Save
    output_path: str = "validated_dataset.json"


def validate_one_sample(
    sample: dict, answer_key: str = "answer", solution_key: str = "generations"
) -> list[bool]:
    """
    Validate whether a question has a verifiable answer with math_verify.
    """
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )

    answer = "\\boxed{" + sample[answer_key] + "}"

    solutions = sample[solution_key]
    if not isinstance(solutions, list):
        solutions = [solutions]

    scores = []
    for solution in solutions:
        ret_score = 0.0

        # math_verify can't handle multiple \boxed{}
        if cli_config.separator in solution:
            solution = solution.split(cli_config.separator)[-1].strip()

        try:
            ret_score, _ = verify_func([answer], [solution])
        except Exception as e:
            print(f"Error in verifying: {e}")
            ret_score = 0
        except TimeoutException:
            ret_score = 0

        scores.append(ret_score)

    return scores


def main(cli_config):
    # Sanitize output path
    output_dir = os.path.dirname(cli_config.output_path)
    if output_dir != "" and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    if os.path.exists(cli_config.dataset_name) and os.path.isdir(
        cli_config.dataset_name
    ):
        samples = datasets.load_from_disk(cli_config.dataset_name)[cli_config.split]
    else:
        samples = datasets.load_dataset(cli_config.dataset_name, split=cli_config.split)

    # Validate samples
    validated_samples = []
    for sample in tqdm(samples, desc="Validating samples", total=len(samples)):
        scores = validate_one_sample(
            sample,
            answer_key=cli_config.answer_key,
            solution_key=cli_config.solution_key,
        )
        if any(scores):
            validated_sample = {
                "problem": sample["problem"],
                "answer": sample["answer"]
            }
            validated_samples.append(validated_sample)

    # Save filtered dataset
    print(f"Saved {len(validated_samples)} valid samples to {cli_config.output_path}")
    with open(cli_config.output_path, "w", encoding="utf8") as f:
        for i in range(len(validated_samples)):
            json.dump(validated_samples[i], f, ensure_ascii=False)
            f.write("\n")    


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    main(cli_config)
