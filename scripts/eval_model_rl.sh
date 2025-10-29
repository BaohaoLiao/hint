#!/bin/bash

# CKPT directory
model_name=Qwen2.5-Math-1.5B
project_name=hint
hint_level=1
experiment_name=Hint${hint_level}_${model_name}
ckpts_dir=/mnt/nushare2/data/baliao/hint/02_rl_level/${experiment_name}
data_path="/mnt/nushare2/data/baliao/dynamic_filter/data"

# Configuration
K=16
GPUS=(0 1 2 3 4 5 6 7)

# Model and dataset arrays
models=()
for step in $(seq 10 10 100); do
    models+=("${ckpts_dir}/global_step_${step}")
done
datasets=("math500" "minerva_math" "olympiadbench" "aime_hmmt_brumo_cmimc_amc23")

# Loop through models and datasets
for model_name in "${models[@]}"; do
    # Convert model to huggingface format
    echo "Converting model: $model_name to huggingface format..."
    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir ${model_name}/actor \
        --target_dir ${model_name}/merged

    echo "Testing model: $model_name"
    for dataset in "${datasets[@]}"; do
        echo "Testing dataset: $dataset"
        
        # Create model/dataset specific output directory
        output_dir=${model_name}/${dataset}
        mkdir -p ${output_dir}
        echo "Output directory: $output_dir"
        
        # Generate data in parallel
        echo "Starting parallel data generation..."
        for ((i=0; i<${#GPUS[@]}; i++)); do
            CUDA_VISIBLE_DEVICES=$i python3 eval/gen_data.py \
                --local_index ${i} \
                --my_world_size ${#GPUS[@]} \
                --model_name_or_path ${model_name}/merged \
                --max_input_length 4096 \
                --output_dir ${output_dir} \
                --K $K \
                --dataset_name_or_path ${data_path}/${dataset} &
        done
        
        # Wait for all parallel processes to complete
        wait
        echo "Data generation completed."
        
        # Merge the generated data
        echo "Merging data..."
        python3 eval/merge_data.py \
            --base_path ${output_dir} \
            --output_dir ${output_dir}/merged_data.jsonl \
            --num_datasets ${#GPUS[@]}
        
        if [ $? -ne 0 ]; then
            echo "Error: Failed to merge data for $model_name on $dataset"
            continue
        fi
        
        # Compute scores
        echo "Computing scores..."
        python3 eval/compute_score.py \
            --dataset_path ${output_dir}/merged_data.jsonl \
            --record_path ${output_dir}/record.txt
        
        if [ $? -ne 0 ]; then
            echo "Error: Failed to compute scores for ${model_name} on ${dataset}"
            continue
        fi
        
        echo "Completed evaluation for ${model_name} on ${dataset}"
        echo "Results saved to: ${output_dir}/record.txt"
        echo "----------------------------------------"
    done
done

echo "All evaluations completed!"
