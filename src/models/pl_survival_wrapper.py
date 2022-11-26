from typing import Tuple, List
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
        self, batch: Tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        image, tabular, event_indicator, days_to_event = batch

        pred_risk = self.model(image, tabular)
        loss = self.loss(pred_risk, event_indicator, days_to_event)

        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: Tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        image, tabular, event_indicator, days_to_event = batch

        pred_risk = self.model(image, tabular)
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
        self.log("val_c_index", c_index)
        return loss

    def test_step(
        self, batch: Tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        image, tabular, event_indicator, days_to_event = batch

        pred_risk = self.model(image, tabular)

        return pred_risk, event_indicator, days_to_event

    def test_epoch_end(self, batches: List[Tuple[Tensor, Tensor, Tensor]]) -> None:
        pred_risk = torch.cat([batch[0] for batch in batches])
        event_indicator = torch.cat([batch[1] for batch in batches])
        days_to_event = torch.cat([batch[2] for batch in batches])

        (
            c_index,
            n_concordant,
            n_discordant,
            n_tied_risk,
            n_tied_time,
        ) = concordance_index_censored(
            event_indicator, days_to_event, pred_risk.squeeze()
        )

        self.log("test_c_index", c_index)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer
