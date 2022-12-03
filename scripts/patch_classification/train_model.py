import pytorch_lightning as pl

from src.utils import train_model, get_data_split
from src.models.pretrained_classification_model import ImgClassificationModel
from src.patch_dataset import PatchDataset

pl.seed_everything(42)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--val-batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--max-epochs", type=int)
    args = parser.parse_args()

    train_df, val_df = get_data_split(args.data_path, test_size=0.1)
    train_dataset = PatchDataset(train_df)
    val_dataset = PatchDataset(val_df)

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
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        log_path=args.log_path,
        max_epochs=args.max_epochs
    )
