from utils import train_model
from models.resnet50 import ResNet50
from models.pretrained_classification_model import ImgClassificationModel
from torchvision.models import VGG19_Weights, vgg19
import pytorch_lightning as pl

pl.seed_everything(42)

# ResNet50
# train_model(ResNet50(), "CRC-VAL-HE-7K")


# VGG-19 or any different pre-trained image classification model from torchvision
if __name__ == "__main__":
    trained_vgg_19 = train_model(
        ImgClassificationModel(vgg19, VGG19_Weights),
        train_batch_size=64,
        data_path="100K-PROCESSED",
        saved_models_path="scratch/saved_models",
    )
