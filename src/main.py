from utils import train_model, load_model
from models.pretrained_classification_model import ImgClassificationModel
from torchvision.models import ResNet50_Weights, resnet50
import pytorch_lightning as pl
from torch import nn

pl.seed_everything(42)

if __name__ == "__main__":
    trained_model = train_model(
        ImgClassificationModel(
            model_name="efficientnet_b0",
            num_classes=9,
            hyperparams={"learning_rate": 1e-3, "momentum": 0.9, "weight_decay": 1e-4},
        ),
        train_batch_size=64,
        data_path="100K-PROCESSED",
        saved_models_path="scratch/saved_models",
    )
