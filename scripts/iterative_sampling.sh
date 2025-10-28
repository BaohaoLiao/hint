#!/bin/bash

cd /data/chatgpt-training-slc-a100/data/baliao/hint/00_data_process/hint

DATA="/mnt/nushare2/data/baliao/hint/data/openr1/sampling_iterative/Qwen3-30B-A3B-Instruct-2507_n4/after_Qwen3-4B-Instruct-2507.json"
MODEL="/mnt/nushare2/data/baliao/PLLMs/qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_NAME="Qwen3-30B-A3B-Instruct-2507"
N=4
MAX_ITERATIONS=32
SAVE_DIR="/mnt/nushare2/data/baliao/hint/data/openr1/sampling_iterative/${MODEL_NAME}_n${N}"
GPUS=(0 1 2 3 4 5 6 7)

mkdir -p ${SAVE_DIR}

# Iterative sampling
for ITER in $(seq 1 ${MAX_ITERATIONS}); do
    echo ""
    echo "========================================"
    echo "Starting iteration ${ITER}/${MAX_ITERATIONS}..."
    echo "========================================"

    # Generate data in parallel across all GPUs
    echo "Launching ${#GPUS[@]} parallel sampling jobs..."
    for ((i=0; i<${#GPUS[@]}; i++)); do
        CUDA_VISIBLE_DEVICES=${GPUS[$i]} python -m data_process.iterative_sampling_extract \
            dataset_path=${DATA} \
            world_size=${#GPUS[@]} \
            local_idx=${i} \
            model_name_or_path=${MODEL} \
            n=${N} \
            iteration=${ITER} \
            max_model_length=18000 \
            max_new_tokens=16384 \
            output_dir=${SAVE_DIR} \
            merged_file="merged_all_iterations.json" &
    done

    # Wait for all sampling jobs to complete
    wait
    echo "All sampling jobs completed for iteration ${ITER}"

    # Merge results from all GPUs
    echo ""
    echo "Merging results from all GPUs..."
    python -m data_process.merge_iterations \
        current_iteration=${ITER} \
        world_size=${#GPUS[@]} \
        output_dir=${SAVE_DIR} \
        merged_file="merged_all_iterations.json"

    MERGE_EXIT_CODE=$?

    # Check exit codes
    if [ ${MERGE_EXIT_CODE} -eq 2 ]; then
        echo ""
        echo "========================================="
        echo "ERROR: Merge failed due to missing files"
        echo "Some GPUs did not complete successfully"
        echo "Stopping pipeline at iteration ${ITER}"
        echo "========================================="
        exit 1
    elif [ ${MERGE_EXIT_CODE} -eq 0 ]; then
        echo ""
        echo "========================================="
        echo "SUCCESS: All problems have passed!"
        echo "Stopping after iteration ${ITER}"
        echo "========================================="
        break
    fi

    echo ""
done

echo ""
echo "========================================="
echo "Iterative sampling complete!"
echo "Final results: ${SAVE_DIR}/merged_all_iterations.json"
echo "========================================="

chown -R 110541254:110541254 ${SAVE_DIR}/..