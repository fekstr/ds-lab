from tkinter import Y
import math

import PIL
import openslide
from torchvision import transforms
import torchstain
import numpy as np


TARGET_MPP = 0.5
TARGET_DIM = 1500


class PreprocessingSVS:
    def __init__(self, image_path, target_path=None) -> None:
        # read slide
        slide = openslide.OpenSlide(image_path)

        # keep only best slide for mpp resampling
        self.scale_factor = float(slide.properties["openslide.mpp-x"]) / TARGET_MPP
        self.slide_dim = slide.dimensions
        level = slide.get_best_level_for_downsample(self.scale_factor)
        self.image = slide.read_region(
            (0, 0), level, slide.level_dimensions[level]
        ).convert("RGB")

        del slide

    def resize_to_target_mpp(self) -> None:
        new_x = math.floor(self.slide_dim[0] / self.scale_factor)
        new_y = math.floor(self.slide_dim[1] / self.scale_factor)
        self.image = self.image.resize((new_x, new_y), PIL.Image.BICUBIC)

    def crop(self) -> None:
        if self.image.size[0] > TARGET_DIM and self.image.size[1] > TARGET_DIM:
            offset_x = round((self.image.size[0] - TARGET_DIM) / 2)
            offset_y = round((self.image.size[1] - TARGET_DIM) / 2)
            self.image = self.image.crop(
                (offset_x, offset_y, offset_x + TARGET_DIM, offset_y + TARGET_DIM)
            )
        else:
            print("WRONG IMAGE DIMENSION... will skip this image")

    def normalise(self) -> None:
        T = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * 255)]
        )
        torch_normaliser = torchstain.normalizers.MacenkoNormalizer(backend="torch")
        target = PIL.Image.open("./Ref.png")
        torch_normaliser.fit(T(target))
        norm, _, _ = torch_normaliser.normalize(I=T(self.image), stains=True)
        self.image = PIL.Image.fromarray(np.uint8(norm.numpy())).convert("RGB")

    def save(self) -> None:
        self.image.save("TCGA-AA-3516.png")


if __name__ == "__main__":
    preprocess = PreprocessingSVS("TCGA-AA-3516.svs")
    preprocess.resize_to_target_mpp()
    preprocess.crop()
    preprocess.normalise()
    preprocess.save()
