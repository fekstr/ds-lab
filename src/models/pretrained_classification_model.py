from typing import Union, Literal

import pytorch_lightning as pl
from torchvision.models import (
    vgg19,
    VGG19_Weights,
    resnet50,
    ResNet50_Weights,
    efficientnet_b0,
    EfficientNet_B0_Weights,
)
import numpy as np
import torch
from torch import nn, optim
from torchvision.models._api import Weights
from torchmetrics.functional.classification import accuracy
from torchmetrics.functional import confusion_matrix

from src.utils import get_heatmap


class ImgClassificationModel(pl.LightningModule):
    def __init__(
        self,
        model_name: Literal["vgg19", "resnet50"],
        num_classes: int,
        hyperparams: dict = {
            "learning_rate": 3e-4,
            "momentum": 0.90,
            "weight_decay": 0.0,
        },
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if model_name == "vgg19":
            self.model = vgg19(VGG19_Weights.DEFAULT)
        elif model_name == "resnet50":
            self.model = resnet50(ResNet50_Weights.DEFAULT)
        elif model_name == "efficientnet_b0":
            self.model = efficientnet_b0(EfficientNet_B0_Weights.DEFAULT)

        self.model_name = model_name
        self.update_output_dim(num_classes)
        self.hyperparams = hyperparams
        self.class_weights = None
        self.num_classes = num_classes

    def training_step(self, batch: tuple, batch_idx: Union[int, list, None]) -> None:
        imgs, labels = batch
        y_hat = self.model(imgs)
        loss = nn.functional.cross_entropy(y_hat, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: tuple, batch_idx: Union[int, list, None]):
        imgs, labels = batch
        y_hat = self.model(imgs)
        loss = nn.functional.cross_entropy(y_hat, labels)
        preds = y_hat.argmax(dim=-1).cpu()
        labels = labels.cpu()
        acc_micro = accuracy(preds, labels, average="micro")
        acc_macro = accuracy(
            preds, labels, average="macro", num_classes=self.num_classes
        )
        self.log("val_acc", acc_micro)
        self.log("val_acc_macro", acc_macro)
        self.log("val_loss", loss)
        return preds, labels

    def validation_epoch_end(self, batches) -> None:
        preds = torch.cat([batch[0] for batch in batches])
        labels = torch.cat([batch[1] for batch in batches])

        acc_worst = accuracy(
            preds, labels, average="none", num_classes=self.num_classes
        ).min()
        self.log("val_acc_worst", acc_worst)

        cm = confusion_matrix(preds, labels, num_classes=self.num_classes)
        cm_norm = confusion_matrix(
            preds, labels, normalize="true", num_classes=self.num_classes
        )
        cm_im = get_heatmap(cm)
        cm_norm_im = get_heatmap(cm_norm)
        self.logger.experiment.add_image(
            "Confusion matrix (validation)", cm_im, self.current_epoch
        )
        self.logger.experiment.add_image(
            "Normalized confusion matrix (validation)", cm_norm_im, self.current_epoch
        )

    def test_step(self, batch: tuple, batch_idx: Union[int, list, None]):
        imgs, labels = batch
        y_hat = self.model(imgs)
        loss = nn.functional.cross_entropy(y_hat, labels)
        preds = y_hat.argmax(dim=-1).cpu()
        labels = labels.cpu()
        acc_micro = accuracy(preds, labels, average="micro")
        acc_macro = accuracy(
            preds, labels, average="macro", num_classes=self.num_classes
        )
        self.log("test_acc", acc_micro)
        self.log("test_acc_macro", acc_macro)
        self.log("test_loss", loss)
        return preds, labels

    def test_epoch_end(self, batches) -> None:
        preds = torch.cat([batch[0] for batch in batches])
        labels = torch.cat([batch[1] for batch in batches])

        acc_worst = accuracy(
            preds, labels, average="none", num_classes=self.num_classes
        ).min()
        self.log("test_acc_worst", acc_worst)

        cm = confusion_matrix(preds, labels, num_classes=self.num_classes)
        cm_norm = confusion_matrix(
            preds, labels, normalize="true", num_classes=self.num_classes
        )
        cm_im = get_heatmap(cm)
        cm_norm_im = get_heatmap(cm_norm)
        self.logger.experiment.add_image(
            "Confusion matrix (test)", cm_im, self.current_epoch
        )
        self.logger.experiment.add_image(
            "Normalized confusion matrix (test)", cm_norm_im, self.current_epoch
        )

    def predict_step(self, batch: tuple, batch_idx, dataloader_idx=0):
        if len(batch) == 1:
            imgs = batch[0]
        else:
            imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        return preds

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.hyperparams["learning_rate"],
            momentum=self.hyperparams["momentum"],
            weight_decay=self.hyperparams["weight_decay"],
        )
        return optimizer

    # Custom methods
    def update_output_dim(self, new_num_classes):
        self.num_classes = new_num_classes
        if self.model_name == "vgg19":
            fc_in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(fc_in_features, new_num_classes)
        elif self.model_name == "resnet50":
            fc_in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(fc_in_features, new_num_classes)
        elif self.model_name == "efficientnet_b0":
            fc_in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(fc_in_features, new_num_classes)

    def set_class_weights(self, class_weights):
        self.class_weights = class_weights
