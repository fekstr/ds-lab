from typing import Tuple
from argparse import ArgumentParser

import torch
from torch import Tensor
from torch import nn
import pytorch_lightning as pl

from sksurv.metrics import concordance_index_censored


class PLSurvivalWrapper(pl.LightningModule):
    def __init__(self, model: nn.Module, loss: nn.Module, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model", "loss"])

        self.model = model
        self.loss = loss

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("PLSurvivalWrapper")
        parser.add_argument("--learning_rate", type=float, default=1e-4)
        parser.add_argument("--weight_decay", type=float, default=1e-5)

    def training_step(
        self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        image, event_indicator, days_to_event = batch

        pred_risk = self.model(image)
        loss = self.loss(pred_risk, event_indicator, days_to_event)

        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        image, event_indicator, days_to_event = batch

        pred_risk = self.model(image)
        loss = self.loss(pred_risk, event_indicator, days_to_event)

        (
            c_index,
            n_concordant,
            n_discordant,
            n_tied_risk,
            n_tied_time,
        ) = concordance_index_censored(
            event_indicator, days_to_event, pred_risk.squeeze()
        )

        self.log("val_loss", loss)
        self.log("C-index", c_index)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer
