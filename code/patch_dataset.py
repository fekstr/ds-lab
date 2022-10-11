import pathlib
from typing import Tuple
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize


class PatchDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        transform=Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ) -> None:
        self.path = pathlib.Path(dataset_path)
        self.classes_folder = sorted(self.path.glob("*"))
        self.transform = transform
        print(self.path)
        self.classes_map = {}
        self.imgs_paths = []

        for i, dir in enumerate(self.classes_folder):
            self.classes_map[dir.name] = i
            self.imgs_paths += [self.path.joinpath(x) for x in dir.glob("*")]

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx) -> Tuple[Tensor, int]:

        img_path = self.imgs_paths[idx]

        image = Image.open(img_path)
        image = ToTensor()(image)
        if self.transform:
            image = self.transform(image)

        return image, self.classes_map[img_path.parents[0].name]
