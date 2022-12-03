#!/bin/bash

#SBATCH -n 4
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=2048
#SBATCH --output=%j.out

logpath="logs/${SLURM_JOB_ID}"
mkdir $logpath

cmd="python -m scripts.patch_classification.test_model \
--log-path $logpath \
--test-batch-size 64 \
$@"

# Save python command to keep track of experiment parameters
echo $cmd > $logpath/command.txt
eval $cmd

mv ${SLURM_JOB_ID}.out $logpath/log.out