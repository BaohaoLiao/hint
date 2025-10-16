#######################################################################
# Validate whether a question has a verifiable answer with math_verify.
# Filter out samples that cannot be verified.
#######################################################################

import os
import chz
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
    output_dir: str = "validated_dataset"


def validate_one_sample(
    sample: dict, answer_key: str = "answer", solution_key: str = "generations"
) -> bool:
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

    if any(scores):
        return True
    return False


def main(cli_config):
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
        is_valid = validate_one_sample(
            sample,
            answer_key=cli_config.answer_key,
            solution_key=cli_config.solution_key,
        )
        validated_sample = {**sample, "is_valid": is_valid}
        validated_samples.append(validated_sample)

    # Save filtered dataset
    valid_samples = [s for s in validated_samples if s["is_valid"]]
    valid_dataset = datasets.Dataset.from_list(valid_samples)
    valid_dataset.save_to_disk(cli_config.output_dir)
    print(f"Saved {len(valid_dataset)} valid samples to {cli_config.output_dir}")


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    main(cli_config)
