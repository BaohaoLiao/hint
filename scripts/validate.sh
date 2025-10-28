#!/bin/bash

python -m data_process.answer_validate \
    dataset_name="open-r1/OpenR1-Math-220k" \
    split="train" \
    answer_key="answer" \
    solution_key="generations" \
    output_path="/mnt/nushare2/data/baliao/hint/data/openr1/validated.json"