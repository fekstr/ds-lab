from tkinter import Y
import math
from pathlib import Path
from tqdm import tqdm

import PIL

# The path can also be read from a config file, etc.
OPENSLIDE_PATH = (
    r"C:\\Users\\dantg\\openslide-win64-20220811\\openslide-win64-20220811\\bin"
)

import os

if hasattr(os, "add_dll_directory"):
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
            print("WRONG IMAGE DIMENSION... will skip this Image ", self.image_path)

    def normalise(self, target_path="Ref.png") -> None:
        T = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * 255)]
        )
        torch_normaliser = torchstain.normalizers.MacenkoNormalizer(backend="torch")
        target = PIL.Image.open(target_path)
        torch_normaliser.fit(T(target))
        try:
            norm, _, _ = torch_normaliser.normalize(I=T(self.image), stains=True)
            self.image = PIL.Image.fromarray(np.uint8(norm.numpy())).convert("RGB")
        except:
            print("WARNING: Normalisation skipped for Image ", self.image_path)

    def save(self) -> None:
        self.image.save(self.target_path)


if __name__ == "__main__":
    import argparse

    ## big image preprocessing (Daniel)
    preprocess = PreprocessingSVS("TCGA-AA-3516.svs")
    preprocess.resize_to_target_mpp()
    preprocess.crop()
    preprocess.normalise()
    preprocess.save()

    ## 224x224 image preprocessing (Frithiof)
    parser = argparse.ArgumentParser()
    parser.add_argument("--type")
    args = parser.parse_args()

    data_path = os.path.join("data", "NCT-CRC-HE-100K-NONORM")
    target_path = os.path.join("data", "100K-PROCESSED")

    c = args.type
    class_path = os.path.join(data_path, c)
    target_class_path = os.path.join(target_path, c)
    for img in tqdm(os.listdir(class_path)):
        img_path = os.path.join(class_path, img)
        target_img_path = os.path.join(target_class_path, img)

        if os.path.isfile(target_img_path):
            continue
        Path(target_class_path).mkdir(parents=True, exist_ok=True)
        preprocess = PreprocessingSVS(img_path, target_path=target_img_path)
        preprocess.normalise(target_path="code/preprocess_images/Ref.png")
        preprocess.save()
