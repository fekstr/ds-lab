from utils import train_model
from models.resnet50 import ResNet50
import pytorch_lightning as pl

pl.seed_everything(42)
train_model(ResNet50, "CRC-VAL-HE-7K")
