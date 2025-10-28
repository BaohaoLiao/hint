#!/bin/bash

set -x

cd /data/chatgpt-training-slc-a100/data/baliao/hint/00_data_process/hint

project_name=hint
experiment_name=SFT-Qwen2.5-Math-1.5B-619
nproc_per_node=2
train_files=/mnt/nushare2/data/baliao/hint/data/openr1/Qwen2.5-Math-1.5B/training_data/sft_train_619.parquet
model_name_or_path=/mnt/nushare2/data/baliao/PLLMs/qwen/Qwen2.5-Math-1.5B
save_dir=/mnt/nushare2/data/baliao/hint/01_sft
save_path=${save_dir}/${experiment_name}

export WANDB_MODE="offline"
export WANDB_DIR=${save_dir}/${experiment_name}/logs

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=${train_files} \
    data.val_files=${train_files} \
    data.prompt_key='prompt' \
    data.response_key='solution' \
    data.max_length=4096 \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=16 \
    data.truncation='left' \
    optim.lr=1e-5 \
    model.partial_pretrain=${model_name_or_path} \
    trainer.default_local_dir=${save_path} \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.logger=['console','wandb'] \
    trainer.total_training_steps=60 \
    trainer.total_epochs=10 \
    trainer.save_freq=20 \
    use_remove_padding=true