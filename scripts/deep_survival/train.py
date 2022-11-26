import pickle

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from src.survival_dataset import SurvivalDataset
from src.models.deep_survival import DeepSurvivalModel
from src.models.mlp_survival import MLPSurvivalModel
from src.models.pl_survival_wrapper import PLSurvivalWrapper
from src.losses.survival_loss import SurvivalLoss


def get_dataloaders(data_path: str, batch_size: int):
    df_train = pd.read_pickle(data_path)

    df_train, df_val = train_test_split(
        df_train, test_size=0.1, stratify=df_train["vital_status"]
    )

    train_dataset = SurvivalDataset(df_train)
    val_dataset = SurvivalDataset(df_val)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
    )
    return train_dataloader, val_dataloader


if __name__ == "__main__":
    from argparse import ArgumentParser

    pl.seed_everything(42)

    parser = ArgumentParser()
    PLSurvivalWrapper.add_model_specific_args(parser)
    pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--batch_size", type=int)
    args = parser.parse_args()

    train_dataloader, val_dataloader = get_dataloaders(args.data_path, args.batch_size)

    dict_args = vars(args)
    model = PLSurvivalWrapper(DeepSurvivalModel(), SurvivalLoss(), **dict_args)

    checkpoint_cb = ModelCheckpoint(
        save_top_k=3, monitor="val_loss", mode="min", save_last=True
    )
    early_stopping_cb = EarlyStopping(monitor="val_loss", mode="min")
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[early_stopping_cb, checkpoint_cb]
    )

    trainer.fit(
        model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
