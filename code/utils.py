from typing import Tuple
import torch
import pathlib
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from statistics import mode
from patch_dataset import PatchDataset
from models.resnet50 import ResNet50


def train_model(
    model: pl.LightningModule,
    data_path: str = "CRC-VAL-HE-7K",
    train_batch_size: int = 32,
    val_batch_size: int = 64,
    max_epochs: int = 10,
    saved_models_path: str = None,
) -> pl.LightningModule:
    current_path = pathlib.Path(__file__).parent.resolve()
    model = model
    dataset = PatchDataset(current_path.parents[0].joinpath("data").joinpath(data_path))
    train_ds, val_ds = random_split(
        dataset=dataset,
        lengths=[int(0.8 * len(dataset)), int(len(dataset) - (0.8 * len(dataset)))],
    )
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False)

    if saved_models_path is None:
        saved_models_path = pathlib.Path(__file__).parents[1].joinpath("saved_models")

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


def load_model(
    model: torch.nn.Module,
    pl_model: pl.LightningModule,
    checkpoint_path: str,
) -> Tuple[pl.LightningModule, pl.Trainer]:

    use_gpu = torch.cuda.is_available()

    model = pl_model.load_from_checkpoint(checkpoint_path, model=model)

    trainer = pl.Trainer(
        accelerator="gpu" if use_gpu else "cpu",
        devices=-1 if use_gpu else None,
    )

    return model, trainer
