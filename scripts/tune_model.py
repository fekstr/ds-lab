import pytorch_lightning as pl

from src.utils import train_model
from src.models.pretrained_classification_model import ImgClassificationModel
from src.patch_dataset import PatchDataset

pl.seed_everything(42)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--tune-data-path")
    parser.add_argument("--saved-models-path")
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--val-batch-size", type=int)
    parser.add_argument("--num-classes", type=int)
    args = parser.parse_args()

    dataset = PatchDataset(args.tune_data_path)
    pretrained_model = ImgClassificationModel.load_from_checkpoint(args.checkpoint_path)
    pretrained_model.update_output_dim(args.num_classes)
    pretrained_model.set_class_weights(dataset.class_weights)

    trained_model = train_model(
        pretrained_model,
        dataset=dataset,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        saved_models_path=args.saved_models_path,
    )
