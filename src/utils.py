from typing import Tuple
import os
import io

import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from src.patch_dataset import PatchDataset


def train_model(
    model: pl.LightningModule,
    train_dataset: PatchDataset,
    val_dataset: PatchDataset,
    log_path: str,
    train_batch_size: int = 32,
    val_batch_size: int = 64,
    max_epochs: int = 10,
    use_weighted_sampling=False,
) -> pl.LightningModule:
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        sampler=train_dataset.weighted_sampler if use_weighted_sampling else None,
    )
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

    if log_path is None:
        raise ValueError("log_path is a required argument")

    use_gpu = torch.cuda.is_available()

    checkpoint_cb = ModelCheckpoint(
        save_top_k=3, monitor="val_loss", mode="min", save_last=True
    )
    early_stopping_cb = EarlyStopping(monitor="val_loss", mode="min")
    logger = TensorBoardLogger(save_dir=log_path, name="", version="")

    trainer = pl.Trainer(
        accelerator="gpu" if use_gpu else "cpu",
        devices=-1 if use_gpu else None,
        max_epochs=max_epochs,
        callbacks=[early_stopping_cb, checkpoint_cb],
        default_root_dir=log_path,
        logger=logger,
        log_every_n_steps=10,
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    best_model_path = '/'.join(checkpoint_cb.best_model_path.split('/')[:-1] + ['best.ckpt'])
    os.rename(checkpoint_cb.best_model_path, best_model_path)

    return model


def test_model(
    model: pl.LightningModule,
    test_dataset: PatchDataset,
    log_path: str,
    test_batch_size: int = 64,
) -> pl.LightningModule:
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
    )

    if log_path is None:
        raise ValueError("log_path is a required argument")

    use_gpu = torch.cuda.is_available()

    logger = TensorBoardLogger(save_dir=log_path, name="", version="")

    trainer = pl.Trainer(
        accelerator="gpu" if use_gpu else "cpu",
        devices=-1 if use_gpu else None,
        default_root_dir=log_path,
        logger=logger,
        log_every_n_steps=10,
    )

    trainer.test(model, dataloaders=test_loader)

    return model


def load_data(data_path):
    paths = []
    classes = []

    for path, _, files in os.walk(data_path):
        for name in files:
            paths.append(os.path.join(path, name))
            classes.append(path.split("/")[-1])

    return paths, classes


def get_data_split(train_path, test_size=0.1) -> Tuple[pd.DataFrame, pd.DataFrame]:

    train_paths, train_classes = load_data(train_path)

    train_paths, val_paths, train_classes, val_classes = train_test_split(
        train_paths, train_classes, stratify=train_classes, test_size=test_size
    )

    train_df = pd.DataFrame({"path": train_paths, "class": train_classes})
    val_df = pd.DataFrame({"path": val_paths, "class": val_classes})

    return train_df, val_df


def get_heatmap(confusion_matrix: torch.Tensor) -> torch.Tensor:
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(left=0.05, right=0.65)
    sns.set(font_scale=1.2)
    sns.heatmap(confusion_matrix.cpu(), annot=True, annot_kws={"size": 16}, ax=ax)
    ax.set_xlabel("Pred")
    ax.set_ylabel("Actual")
    buf = io.BytesIO()
    plt.savefig(buf, format="jpeg", bbox_inches="tight")
    buf.seek(0)
    im = Image.open(buf)
    im = torchvision.transforms.ToTensor()(im)
    plt.close(fig)
    return im
