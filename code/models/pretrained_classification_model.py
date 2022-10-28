from typing import Union
import pytorch_lightning as pl
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn, optim
from torchvision.models._api import Weights


class ImgClassificationModel(pl.LightningModule):
    def __init__(self, model: nn.Module, model_weights: Weights = None) -> None:
        super().__init__()
        if model_weights is not None:
            weights = model_weights.DEFAULT
        else:
            weights = None
        self.model = model(weights=weights)

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
        preds = y_hat.argmax(dim=-1)
        acc = (labels == preds).float().mean()
        self.log("val_acc", acc)
        self.log("val_loss", loss)

    def test_step(self, batch: tuple, batch_idx: Union[int, list, None]):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        self.log("test_acc", acc)

    def predict_step(self, batch: tuple, batch_idx, dataloader_idx=0):
        if len(batch) == 1:
            imgs = batch[0]
        else:
            imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        return preds

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=3e-4, momentum=0.90)
        return optimizer
