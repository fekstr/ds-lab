#!/bin/bash

#SBATCH -n 8
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=2048

python -m scripts.train_model