#!/bin/bash

#SBATCH -n 8
#SBATCH --gpus=4
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=2048
#SBATCH --output=slurm_logs/slurm-%j.out

python -m scripts.train_model \
--model-name efficientnet_b0 \
--data-path data/100K-PROCESSED \
--saved-models-path scratch/saved_models \
--train-batch-size 64 \
--val-batch-size 64 \
--num-classes 9 \
--lr 0.001 \
--momentum 0.9 \
--weight-decay 0.0001