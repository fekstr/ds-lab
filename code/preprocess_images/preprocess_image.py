from tkinter import Y
import math

import PIL
# The path can also be read from a config file, etc.
OPENSLIDE_PATH = r'C:\\Users\\dantg\\openslide-win64-20220811\\openslide-win64-20220811\\bin'

import os
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
from torchvision import transforms
import torchstain
import numpy as np


TARGET_MPP = 0.5
TARGET_DIM = 1500


class PreprocessingSVS:
    def __init__(self, image_path, target_path=None) -> None:
        self.image_path = image_path
        if not target_path:
            self.target_path = image_path.split(".")[0] + "_processed.tif"
        else:
            self.target_path = target_path

        # read slide
        if image_path.split(".")[-1] == "svs":
            self.if_svs = True
            slide = openslide.OpenSlide(image_path)

            # keep only best slide for mpp resampling
            self.scale_factor = float(slide.properties["openslide.mpp-x"]) / TARGET_MPP
            self.image_dim = slide.dimensions
            level = slide.get_best_level_for_downsample(self.scale_factor)
            self.image = slide.read_region(
                (0, 0), level, slide.level_dimensions[level]
            ).convert("RGB")
            del slide

        else:
            self.if_svs = False
            self.image = PIL.Image.open(image_path)
            self.image_dim = self.image.size

    def resize_to_target_mpp(self) -> None:
        new_x = math.floor(self.image_dim[0] * self.scale_factor)
        new_y = math.floor(self.image_dim[1] * self.scale_factor)
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
        self.image.save(self.target_path)


if __name__ == "__main__":

    ## big image preprocessing (Daniel)
    preprocess = PreprocessingSVS("TCGA-AA-3516.svs")
    preprocess.resize_to_target_mpp()
    preprocess.crop()
    preprocess.normalise()
    preprocess.save()

    ## 224x224 image preprocessing (Frithiof)
    preprocess = PreprocessingSVS("ADI-TCGA-AAICEQFN.tif")
    preprocess.normalise()
    preprocess.save()