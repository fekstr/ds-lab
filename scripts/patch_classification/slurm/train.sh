#!/bin/bash

#SBATCH -n 4
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2048
#SBATCH --output=%j.out

logpath="logs/${SLURM_JOB_ID}"
mkdir $logpath

cmd="python -m scripts.patch_classification.train_model \
--log-path logs/${SLURM_JOB_ID} \
--train-batch-size 64 \
--val-batch-size 64 \
--lr 0.001 \
--momentum 0.9 \
--weight-decay 0.0001 \
--max-epochs 10 \
$@"

# Save python command to keep track of experiment parameters
echo $cmd > $logpath/command.txt
eval $cmd

mv ${SLURM_JOB_ID}.out $logpath/log.out