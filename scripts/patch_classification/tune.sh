#!/bin/bash

#SBATCH -n 4
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=2048

python -m scripts.tune_model \
--checkpoint-path scratch/saved_models/lightning_logs/version_1884922/checkpoints/epoch=9-step=3130.ckpt \
--tune-data-path data/PATH-DT-MSU-TRAIN \
--saved-models-path scratch/saved_models \
--train-batch-size 64 \
--val-batch-size 64 \
--num-classes 5