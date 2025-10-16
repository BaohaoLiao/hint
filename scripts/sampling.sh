#!/bin/bash

DATA="/mnt/nushare2/data/baliao/hint/data/openr1/validated.json"
MODEL="/mnt/nushare2/data/baliao/PLLMs/qwen/Qwen2.5-Math-1.5B"
MODEL_NAME="Qwen2.5-Math-1.5B"
N=8
SAVE_DIR="/mnt/nushare2/data/baliao/hint/data/openr1/sampling/${MODEL_NAME}_n${N}" 
GPUS=(0 1 2 3 4 5 6 7)

# Generate data in parallel
echo "Starting parallel data generation..."
for ((i=0; i<${#GPUS[@]}; i++)); do
    CUDA_VISIBLE_DEVICES=${i} python -m data_process.sampling \
        dataset_path=${DATA} \
        wolrd_size=${#GPUS[@]} \
        local_idx=${i} \
        model_name=${MODEL} \
        n=${N} \
        output_dir=${SAVE_DIR} &
done

# Wait for all jobs to complete
wait

# Build the file list dynamically
echo "Merging output files..."
FILE_LIST=""
for ((i=0; i<${#GPUS[@]}; i++)); do
    FILE_LIST="${FILE_LIST} ${SAVE_DIR}/${i}.json"
done

cat ${FILE_LIST} > ${SAVE_DIR}/merged.json

# Compute score
echo "Computing scores..."
python -m data_process.compute_score \
    dataset_path=${SAVE_DIR}/merged.json \