### EFFICIENTNET_B0 MODELS ###

# Train and test model trained from scratch on NCT-CRC-HE-100K
efficientnet_train_nct_id=$(sbatch --parsable \
scripts/patch_classification/slurm/train.sh \
--model-name efficientnet_b0 \
--data-path data/NCT-CRC-HE-100K-PROCESSED \
--num-classes 9)
efficientnet_test_scratch_nct_id=$(sbatch --parsable --dependency=afterok:${efficientnet_train_nct_id} \
scripts/patch_classification/slurm/test.sh \
--checkpoint-path logs/${efficientnet_train_nct_id}/checkpoints/best.ckpt \
--test-data-path data/CRC-VAL-HE-7K-PROCESSED \
--num-classes 9)

# Tune the model trained on NCT-CRC-HE-100K on PATH-DT-MSU-TRAIN and test the tuned model
efficientnet_tune_path_id=$(sbatch --parsable --dependency=afterok:${efficientnet_train_nct_id} \
scripts/patch_classification/slurm/tune.sh \
--checkpoint-path logs/${efficientnet_train_nct_id}/checkpoints/best.ckpt \
--tune-data-path data/PATH-DT-MSU-TRAIN \
--num-classes 5)
efficientnet_test_tuned_id=$(sbatch --parsable --dependency=afterok:${efficientnet_tune_path_id} \
scripts/patch_classification/slurm/test.sh \
--checkpoint-path logs/${efficientnet_tune_path_id}/checkpoints/best.ckpt \
--test-data-path data/PATH-DT-MSU-TEST \
--num-classes 5)

# Train and test model trained from scratch on PATH-DT-MSU-TRAIN
efficientnet_train_path_id=$(sbatch --parsable \
scripts/patch_classification/slurm/train.sh \
--model-name efficientnet_b0 \
--data-path data/PATH-DT-MSU-TRAIN \
--num-classes 5)
efficientnet_test_scratch_path_id=$(sbatch --parsable --dependency=afterok:${efficientnet_train_path_id} \
scripts/patch_classification/slurm/test.sh \
--checkpoint-path logs/${efficientnet_train_path_id}/checkpoints/best.ckpt \
--test-data-path data/PATH-DT-MSU-TEST \
--num-classes 5)

### RESNET50 MODELS ###

# Train and test model trained from scratch on NCT-CRC-HE-100K
resnet_train_nct_id=$(sbatch --parsable \
scripts/patch_classification/slurm/train.sh \
--model-name resnet50 \
--data-path data/NCT-CRC-HE-100K-PROCESSED \
--num-classes 9)
resnet_test_scratch_nct_id=$(sbatch --parsable --dependency=afterok:${resnet_train_nct_id} \
scripts/patch_classification/slurm/test.sh \
--checkpoint-path logs/${resnet_train_nct_id}/checkpoints/best.ckpt \
--test-data-path data/CRC-VAL-HE-7K-PROCESSED \
--num-classes 9)

# Tune the model trained on NCT-CRC-HE-100K on PATH-DT-MSU-TRAIN and test the tuned model
resnet_tune_path_id=$(sbatch --parsable --dependency=afterok:${resnet_train_nct_id} \
scripts/patch_classification/slurm/tune.sh \
--checkpoint-path logs/${resnet_train_nct_id}/checkpoints/best.ckpt \
--tune-data-path data/PATH-DT-MSU-TRAIN \
--num-classes 5)
resnet_test_tuned_id=$(sbatch --parsable --dependency=afterok:${resnet_tune_path_id} \
scripts/patch_classification/slurm/test.sh \
--checkpoint-path logs/${resnet_tune_path_id}/checkpoints/best.ckpt \
--test-data-path data/PATH-DT-MSU-TEST \
--num-classes 5)

# Train and test model trained from scratch on PATH-DT-MSU-TRAIN
resnet_train_path_id=$(sbatch --parsable \
scripts/patch_classification/slurm/train.sh \
--model-name resnet50 \
--data-path data/PATH-DT-MSU-TRAIN \
--num-classes 5)
resnet_test_scratch_path_id=$(sbatch --parsable --dependency=afterok:${resnet_train_path_id} \
scripts/patch_classification/slurm/test.sh \
--checkpoint-path logs/${resnet_train_path_id}/checkpoints/best.ckpt \
--test-data-path data/PATH-DT-MSU-TEST \
--num-classes 5)

### VGG19 MODELS ###

# Train and test model trained from scratch on NCT-CRC-HE-100K
vgg_train_nct_id=$(sbatch --parsable \
scripts/patch_classification/slurm/train.sh \
--model-name vgg19 \
--data-path data/NCT-CRC-HE-100K-PROCESSED \
--num-classes 9)
vgg_test_scratch_nct_id=$(sbatch --parsable --dependency=afterok:${vgg_train_nct_id} \
scripts/patch_classification/slurm/test.sh \
--checkpoint-path logs/${vgg_train_nct_id}/checkpoints/best.ckpt \
--test-data-path data/CRC-VAL-HE-7K-PROCESSED \
--num-classes 9)

# Tune the model trained on NCT-CRC-HE-100K on PATH-DT-MSU-TRAIN and test the tuned model
vgg_tune_path_id=$(sbatch --parsable --dependency=afterok:${vgg_train_nct_id} \
scripts/patch_classification/slurm/tune.sh \
--checkpoint-path logs/${vgg_train_nct_id}/checkpoints/best.ckpt \
--tune-data-path data/PATH-DT-MSU-TRAIN \
--num-classes 5)
vgg_test_tuned_id=$(sbatch --parsable --dependency=afterok:${vgg_tune_path_id} \
scripts/patch_classification/slurm/test.sh \
--checkpoint-path logs/${vgg_tune_path_id}/checkpoints/best.ckpt \
--test-data-path data/PATH-DT-MSU-TEST \
--num-classes 5)

# Train and test model trained from scratch on PATH-DT-MSU-TRAIN
vgg_train_path_id=$(sbatch --parsable \
scripts/patch_classification/slurm/train.sh \
--model-name vgg19 \
--data-path data/PATH-DT-MSU-TRAIN \
--num-classes 5)
vgg_test_scratch_path_id=$(sbatch --parsable --dependency=afterok:${vgg_train_path_id} \
scripts/patch_classification/slurm/test.sh \
--checkpoint-path logs/${vgg_train_path_id}/checkpoints/best.ckpt \
--test-data-path data/PATH-DT-MSU-TEST \
--num-classes 5)