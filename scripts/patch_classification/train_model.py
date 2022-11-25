import pytorch_lightning as pl

from src.utils import train_model
from src.models.pretrained_classification_model import ImgClassificationModel
from src.patch_dataset import PatchDataset

pl.seed_everything(42)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name")
    parser.add_argument("--data-path")
    parser.add_argument("--saved-models-path")
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--val-batch-size", type=int)
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--weight-decay", type=float)
    args = parser.parse_args()

    dataset = PatchDataset(args.data_path)
    model = ImgClassificationModel(
        model_name=args.model_name,
        num_classes=args.num_classes,
        hyperparams={
            "learning_rate": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
        },
    )

    trained_model = train_model(
        model=model,
        dataset=dataset,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        saved_models_path=args.saved_models_path,
    )
