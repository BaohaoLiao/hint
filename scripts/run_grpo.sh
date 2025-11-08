#!/bin/bash

set -xeuo pipefail

export VLLM_ATTENTION_BACKEND=XFORMERS
export WORKING_DIR="${PWD}"

# Model
model_name_or_path="/mnt/nushare2/data/baliao/PLLMs/qwen/Qwen2.5-Math-1.5B"
model_name="Qwen2.5-Math-1.5B"
hint_level=1

# Wandb setting
project_name="hint"
experiment_name="Hint${hint_level}_${model_name}"

# Output
ckpts_dir="/mnt/nushare2/data/baliao/hint/02_rl_level/${experiment_name}"
mkdir -p "${ckpts_dir}/logs"

# Trainig setting
NGPUS=8
train_prompt_bsz=64
train_prompt_mini_bsz=16

# Algorithm setting
algorithm=grpo
n=32
kl_coef=0.001
use_kl_in_reward=False
use_kl_loss=True
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28

# Training data
train_path="/mnt/nushare2/data/baliao/hint/data/openr1/Qwen2.5-Math-1.5B/training_data/train_619.parquet"
test_path="/mnt/nushare2/data/baliao/hint/data/openr1/Qwen2.5-Math-1.5B/training_data/train_619.parquet"
train_files="['$train_path']"
test_files="['$test_path']"

export WANDB_MODE="offline"
export WANDB_DIR=${ckpts_dir}/logs


python3 -m verl.trainer.main_ppo \
    data.train_files=${train_files} \
    data.val_files=${test_files} \
    data.prompt_key="prompt_with_hint_level_${hint_level}" \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    algorithm.adv_estimator=${algorithm} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.model.path=${model_name_or_path} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=${NGPUS} \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=50 \
    trainer.default_local_dir=${ckpts_dir} \
    trainer.total_training_steps=100 \
    trainer.total_epochs=1000 2>&1 | tee ${ckpts_dir}/logs/log
    