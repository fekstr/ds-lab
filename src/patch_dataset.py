import os
from typing import Tuple

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


def get_class_weights(n_samples, class_counts, unique_classes):
    class_weights = [1 - class_counts[cl] / n_samples for cl in unique_classes]
    class_weights = torch.tensor(
        class_weights,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32,
    )
    return class_weights


class PatchDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform=Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        augmentation=True,
    ) -> None:
        self.labels = df["class"].to_numpy()
        unique_classes = sorted(df["class"].unique())
        self.class_map = {cl: i for i, cl in enumerate(unique_classes)}
        self.img_paths = df["path"].to_numpy()

        class_counts = df["class"].value_counts()
        class_weights = {cl: 1 / class_counts[cl] for cl in unique_classes}
        sample_weights = [class_weights[cl] for cl in self.labels]

        self.weighted_sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(self.img_paths)
        )

        self.transform = transform

        if augmentation:
            self.augmentation = torch.nn.Sequential(
                RandomVerticalFlip(0.5), RandomHorizontalFlip(0.5)
            )
        else:
            self.augmentation = None

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx) -> Tuple[Tensor, int]:

        img_path = self.img_paths[idx]

        image = Image.open(img_path)
        image = ToTensor()(image)
        assert image.shape == torch.Size([3, 224, 224])
        if self.transform:
            image = self.transform(image)

        if self.augmentation:
            image = self.augmentation(image)

        label = self.labels[idx]

        return image, self.class_map[label]
