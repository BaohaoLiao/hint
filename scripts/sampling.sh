#!/bin/bash

python -m data_process.sampling \
    dataset_name="/mnt/nushare2/data/baliao/hint/data/openr1_validated" \
    split="train" \
    answer_key="answer" \
    solution_key="generations" \
    output_dir="/mnt/nushare2/data/baliao/hint/data/openr1_validated"