#!/bin/bash

#SBATCH -n 4
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=2048
#SBATCH --output=slurm_logs/slurm-%j.out

python -m scripts.test_model \
--checkpoint-path scratch/saved_models/lightning_logs/version_2648177/checkpoints/last.ckpt \
--test-data-path data/PATH-DT-MSU-TEST \
--saved-models-path scratch/saved_models \
--test-batch-size 64 \
--num-classes 5