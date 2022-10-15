from statistics import mode
from patch_dataset import PatchDataset
from models.resnet50 import ResNet50
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pathlib


def train_model(
    model: pl.LightningModule,
    data_path: str = "CRC-VAL-HE-7K",
    train_batch_size: int = 32,
    val_batch_size: int = 64,
    max_epochs: int = 10,
) -> pl.LightningModule:
    current_path = pathlib.Path(__file__).parent.resolve()
    model = model
    dataset = PatchDataset(current_path.parents[0].joinpath("data").joinpath(data_path))
    train_ds, val_ds = random_split(
        dataset=dataset,
        lengths=[int(0.8 * len(dataset)), int(len(dataset) - (0.8 * len(dataset)))],
    )
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False)

    trainer = pl.Trainer(
        gpus=-1,
        max_epochs=max_epochs,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
        default_root_dir=pathlib.Path(__file__).parents[1].joinpath("saved_models"),
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    return model
