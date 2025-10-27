#!/bin/bash

cd /data/chatgpt-training-slc-a100/data/baliao/hint/00_data_process/hint

DATA="/mnt/nushare2/data/baliao/hint/data/openr1/sampling_iterative/hint_data/Qwen3-4B-Instruct-2507_passk_0/samples_with_hints.json"
MODEL="/mnt/nushare2/data/baliao/PLLMs/qwen/Qwen2.5-Math-1.5B"
MODEL_NAME="Qwen2.5-Math-1.5B"
N=8
MAX_ITERATIONS=16
HINT_LEVEL=1
SAVE_DIR="/mnt/nushare2/data/baliao/hint/data/openr1/sampling_iterative/hint_data/Qwen3-4B-Instruct-2507_passk_0/sampling_with_hints/hintlevel${HINT_LEVEL}/${MODEL_NAME}_n${N}"
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
        CUDA_VISIBLE_DEVICES=${GPUS[$i]} python -m data_process.iterative_sampling \
            dataset_path=${DATA} \
            world_size=${#GPUS[@]} \
            local_idx=${i} \
            model_name_or_path=${MODEL} \
            n=${N} \
            iteration=${ITER} \
            output_dir=${SAVE_DIR} \
            merged_file="merged_all_iterations.json" \
            hint_level=${HINT_LEVEL} &
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

chown -R 110541254:110541254 ${SAVE_DIR}