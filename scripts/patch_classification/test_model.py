import pytorch_lightning as pl
import pandas as pd

from src.utils import test_model, load_data
from src.models.pretrained_classification_model import ImgClassificationModel
from src.patch_dataset import PatchDataset

pl.seed_everything(42)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--test-data-path", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--test-batch-size", type=int)
    args = parser.parse_args()

    test_paths, test_classes = load_data(args.test_data_path)
    test_df = pd.DataFrame({"path": test_paths, "class": test_classes})
    test_dataset = PatchDataset(test_df)

    pretrained_model = ImgClassificationModel.load_from_checkpoint(
        args.checkpoint_path, num_classes=args.num_classes
    )

    trained_model = test_model(
        pretrained_model,
        test_dataset=test_dataset,
        test_batch_size=args.test_batch_size,
        log_path=args.log_path,
    )
