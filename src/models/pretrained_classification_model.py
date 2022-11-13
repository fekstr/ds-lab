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
from torch import nn, optim
from torchvision.models._api import Weights


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

    def training_step(self, batch: tuple, batch_idx: Union[int, list, None]) -> None:
        imgs, labels = batch
        y_hat = self.model(imgs)
        loss = nn.functional.cross_entropy(y_hat, labels, weight=self.class_weights)
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
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.hyperparams["learning_rate"],
            momentum=self.hyperparams["momentum"],
            weight_decay=self.hyperparams["weight_decay"],
        )
        return optimizer

    # Custom methods
    def update_output_dim(self, new_num_classes):
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
