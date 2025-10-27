#!/bin/bash

cd /data/chatgpt-training-slc-a100/data/baliao/hint/00_data_process/hint

DATA="/mnt/nushare2/data/baliao/hint/data/openr1/sampling_iterative/filtered_data/Qwen3-4B-Instruct-2507_passk_0.json"
MODEL="azure-chat-completions-gpt-5-nano-2025-08-07-sandbox"
SAVE_DIR="/mnt/nushare2/data/baliao/hint/data/openr1/sampling_iterative/hint_data"

mkdir -p ${SAVE_DIR}

python -m data_process.hint_generation_parallel \
    --dataset-path ${DATA} \
    --model-name ${MODEL} \
    --output-dir ${SAVE_DIR} \
    --checkpoint-interval 10