import pickle

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from src.survival_dataset import SurvivalDataset
from src.models.deep_survival import DeepSurvivalModel
from src.models.mlp_survival import MLPSurvivalModel
from src.models.pl_survival_wrapper import PLSurvivalWrapper
from src.losses.survival_loss import SurvivalLoss


if __name__ == "__main__":
    from argparse import ArgumentParser

    pl.seed_everything(42)

    parser = ArgumentParser()
    PLSurvivalWrapper.add_model_specific_args(parser)
    pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--batch_size", type=int)
    args = parser.parse_args()

    df_test = pd.read_pickle(args.data_path)
    test_dataset = SurvivalDataset(df_test)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
    )

    dict_args = vars(args)
    model = PLSurvivalWrapper.load_from_checkpoint(
        args.checkpoint_path, model=DeepSurvivalModel(), loss=SurvivalLoss()
    )

    trainer = pl.Trainer.from_argparse_args(args)

    trainer.test(model=model, dataloaders=test_dataloader)
