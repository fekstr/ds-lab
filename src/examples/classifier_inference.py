import pathlib

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.patch_dataset import PatchDataset
from src.models.pretrained_classification_model import ImgClassificationModel

if __name__ == "__main__":
    checkpoint_path = pathlib.PosixPath(
        "scratch/saved_models/lightning_logs/version_1884922/checkpoints/epoch=9-step=3130.ckpt"
    )

    model = ImgClassificationModel.load_from_checkpoint(checkpoint_path)
    trainer = pl.Trainer(accelerator="auto")

    test_dataset = PatchDataset(dataset_path="data/CRC-VAL-HE-7K")
    test_loader = DataLoader(test_dataset, batch_size=64)

    preds = trainer.predict(model=model, dataloaders=[test_loader])
