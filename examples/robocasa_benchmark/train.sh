#!/usr/bin/env bash
#SBATCH --job-name=gr00t_robocasa_seen_pretrain
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

torchrun --nproc_per_node=2 --master_port=29500 \
    scripts/gr00t_finetune_soup.py \
    --dataset_soup pretrain_atomic_seen \
    --base_model_path nvidia/GR00T-N1.7-3B \
    --embodiment_tag ROBOCASA_PANDA_OMRON \
    --modality_config_path examples/robocasa_benchmark/modality_config.py \
    --output_dir ./checkpoints/robocasa_seen_pretrain \
    --num_gpus 2 \
    --global_batch_size 128 \
    --max_steps 60000 \
    --save_steps 2000 \
    --use_wandb
