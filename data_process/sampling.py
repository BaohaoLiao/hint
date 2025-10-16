#######################################################################
# Sampling generations from a dataset
#######################################################################

import os
import chz

import datasets

@chz.chz
class CLIConfig:
    # dataset
    dataset_name: str = "open-r1/OpenR1-Math-220k"
    split: str = "train"

    # model
    model_name_or_path: str = "Qwen/Qwen2.5-Math-1.5B"

    # sampling
    n: int = 16
    num_samples: int = -1  # -1 means all samples
    random_seed: int = 42

    # Save
    output_path: str = 


def main(cli_config):
    # Load dataset
    if os.path.exists(cli_config.dataset_name) and os.path.isdir(
        cli_config.dataset_name
    ):
        samples = datasets.load_from_disk(cli_config.dataset_name)[cli_config.split]
    else:
        samples = datasets.load_dataset(cli_config.dataset_name, split=cli_config.split)

    # Load model





if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    main(cli_config)
