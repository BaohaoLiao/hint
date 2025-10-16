#######################################################################
# Validate whether a question has a verifiable answer with math_verify.
# Filter out samples that cannot be verified.
#######################################################################

import re
import os
import chz
from tqdm import tqdm

import datasets
from math_verify.metric import math_metric
from math_verify.errors import TimeoutException
from math_verify.parser import LatexExtractionConfig


@chz.chz
class CLIConfig:
    # dataset
    dataset_name: str = "open-r1/OpenR1-Math-220k"
    split: str = "default"
    answer_key: str = "answer"
    solution_key: str = "generations"

    # Save
    output_dir: str = "validated_dataset"


def extract_boxed(text: str) -> str:
    """
    Extract the context of the last \\boxed{...} in the text.
    """
    boxed_strs = []
    stack = []
    for ichar in range(len(text)):
        if text[ichar] == "{":
            stack.append(ichar)
        elif text[ichar] == "}":
            if len(stack) == 0:
                raise ValueError("Unmatched }")
            last_open_start = stack.pop()
            # check if start is preceded by \boxed
            if text[:last_open_start].endswith("\\boxed"):
                boxed_strs.append(text[last_open_start + 1 : ichar])
    if len(boxed_strs) > 0:
        return boxed_strs[-1]
    else:
        # maybe there's something like '\boxed 2' without curly braces
        match = re.search(r"\\boxed\s+([a-zA-Z0-9]+)", text)
        if match:
            return match.group(1)
        else:
            raise ValueError("No boxed strings found")
        

def validate_one_sample(
        sample: dict, 
        answer_key: str = "answer", 
        solution_key: str = "generations"
    ) -> bool:
    """ 
    Validate whether a question has a verifiable answer with math_verify.
    """
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig()),
        pred_extraction_target=(LatexExtractionConfig()),
    )

    answer = "\\boxed{" + sample[answer_key] + "}"

    solutions = sample[solution_key]
    if not isinstance(solutions, list):
        solutions = [solutions]

    scores = []
    for solution in solutions:
        ret_score = 0.0
        pred = extract_boxed(solution)

        if not pred:
            continue

        pred = "\\boxed{" + pred + "}"
        try:
            ret_score, _ = verify_func([answer], [pred])
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
    if os.path.exists(cli_config.dataset_name) and os.path.isdir(cli_config.dataset_name):
        samples = datasets.load_from_disk(cli_config.dataset_name)[cli_config.split]
    else:
        samples = datasets.load_dataset(cli_config.dataset_name, split=cli_config.split)

    # Validate samples
    for sample in tqdm(samples, desc="Validating samples", total=len(samples)):
        sample["is_valid"] = validate_one_sample(
            sample, 
            answer_key=cli_config.answer_key, 
            solution_key=cli_config.solution_key
        )

    # Save filtered dataset
    valid_samples = [s for s in samples if s["is_valid"]]
    valid_dataset = datasets.Dataset.from_list(valid_samples)
    valid_dataset.save_to_disk(cli_config.output_dir)
    print(f"Saved {len(valid_dataset)} valid samples to {cli_config.output_dir}")


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    main(cli_config)