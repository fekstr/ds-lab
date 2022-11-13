import pathlib
import os
from typing import Tuple
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import (
    ToTensor,
    Normalize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)
import torch


class PatchDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        transform=Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        augmentation=True,
    ) -> None:
        self.path = pathlib.Path(dataset_path)
        self.class_paths = sorted(list(self.path.iterdir()))
        self.transform = transform
        self.classes_map = {}
        self.imgs_paths = []

        if augmentation:
            self.augmentation = torch.nn.Sequential(
                RandomVerticalFlip(0.5), RandomHorizontalFlip(0.5)
            )
        else:
            self.augmentation = None

        for i, class_path in enumerate(self.class_paths):
            self.classes_map[class_path.name] = i
            self.imgs_paths += list(class_path.iterdir())

        class_counts = {}
        for cl in self.classes_map.keys():
            cl_path = os.path.join(dataset_path, cl)
            count = len(os.listdir(cl_path))
            class_counts[cl] = count

        n_samples = len(self.imgs_paths)
        self.class_weights = [1 - count / n_samples for count in class_counts.values()]
        self.class_weights = torch.tensor(
            self.class_weights, device="cuda" if torch.cuda.is_available() else "cpu"
        )

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx) -> Tuple[Tensor, int]:

        img_path = self.imgs_paths[idx]

        image = Image.open(img_path)
        image = ToTensor()(image)
        assert image.shape == torch.Size([3, 224, 224])
        if self.transform:
            image = self.transform(image)

        if self.augmentation:
            image = self.augmentation(image)

        return image, self.classes_map[img_path.parents[0].name]
