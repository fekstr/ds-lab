from torchvision.models import vgg19
from torch.utils.data import DataLoader

from utils import load_model
from patch_dataset import PatchDataset
from models.pretrained_classification_model import ImgClassificationModel

if __name__ == "__main__":
    # 1. Run `sh scripts/download_checkpoint.sh` to download the model weights
    # 2. Update paths if necessary (they should be relative to "code")

    model, trainer = load_model(
        model=vgg19,
        pl_model=ImgClassificationModel,
        checkpoint_path="../checkpoint.ckpt",
    )

    test_dataset = PatchDataset(dataset_path="CRC-VAL-HE-7K")
    test_loader = DataLoader(test_dataset, batch_size=64)

    preds = trainer.predict(model=model, dataloaders=[test_loader])
