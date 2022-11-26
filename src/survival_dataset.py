import os
from typing import Tuple
import random

import pandas as pd
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.transforms import (
    ToTensor,
    Normalize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)
import torch
from torchvision.models import ResNet50_Weights

X_cols = [
    "ADI",
    "BACK",
    "DEB",
    "LYM",
    "MUC",
    "MUS",
    "NORM",
    "STR",
    "TUM",
    "years_to_birth",
    "gender",
]


class SurvivalDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform=ResNet50_Weights.DEFAULT.transforms(),
        augmentation=True,
    ) -> None:
        self.X = df.loc[:, X_cols].to_numpy()
        self.event_indicator = df["vital_status"].to_numpy()
        self.days_to_event = df["days_to_event"].to_numpy()
        self.patient_ids = df.index.to_numpy()
        self.patient_slide_map = df.loc[:, "slide_paths"].to_dict()

        self.transform = transform

        if augmentation:
            self.augmentation = torch.nn.Sequential(
                RandomVerticalFlip(0.5), RandomHorizontalFlip(0.5)
            )
        else:
            self.augmentation = None

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        patient_id = self.patient_ids[idx]

        img_path = random.choice(self.patient_slide_map[patient_id])

        image = Image.open(img_path)
        image = ToTensor()(image)
        assert image.shape == torch.Size([3, 1500, 1500])
        if self.transform:
            # TODO: update transform to use full image
            image = self.transform(image)

        if self.augmentation:
            image = self.augmentation(image)

        event_indicator = torch.tensor(self.event_indicator[idx])
        days_to_event = torch.tensor(self.days_to_event[idx])
        tabular = torch.tensor(self.X[idx]).float()

        return image, tabular, event_indicator, days_to_event


# Example usage
if __name__ == "__main__":
    import pickle
    import os

    import pandas as pd

    with open("data/TCGA_train.pkl", "rb") as f:
        train_df = pickle.load(f)

    dataset = SurvivalDataset(train_df)
