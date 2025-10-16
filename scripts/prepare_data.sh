#!/bin/bash

python -m data_process.answer_validate \
    --dataset_name "open-r1/OpenR1-Math-220k" \
    --split "default" \
    --answer_key "answer" \
    --solution_key "generations" \
    --output_dir "/data/chatgpt-training-slc-a100/data/baliao/hint/data/openr1_validated"