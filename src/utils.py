from typing import Tuple
import torch
import pathlib
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np

from statistics import mode
from src.patch_dataset import PatchDataset
from src.models.resnet50 import ResNet50


def train_model(
    model: pl.LightningModule,
    data_path: str,
    saved_models_path: str,
    train_batch_size: int = 32,
    val_batch_size: int = 64,
    max_epochs: int = 10,
) -> pl.LightningModule:
    dataset = PatchDataset(data_path)
    train_ds, val_ds = random_split(
        dataset=dataset,
        lengths=[int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))],
    )
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False)

    if saved_models_path is None:
        raise ValueError("saved_models_path is a required argument")

    use_gpu = torch.cuda.is_available()

    trainer = pl.Trainer(
        accelerator="gpu" if use_gpu else "cpu",
        devices=-1 if use_gpu else None,
        max_epochs=max_epochs,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
        default_root_dir=saved_models_path,
        log_every_n_steps=10,
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    return model


def specificity(y_true, y_pred):
    TN = np.sum(np.logical_and(y_pred == 0, y_true == 0))

    N = len(y_true) - np.sum(y_true)

    return TN / N
