import os

import numpy as np
import openslide
import matplotlib
from segmentation_mask_overlay import overlay_masks
import PIL
import math

LEVEL = 0


class plotSegmentation:
    def __init__(
        self,
        original_folder: str,
        source_folder: str,
        target_folder: str,
    ):
        self.original_folder = original_folder
        self.source_folder = source_folder
        self.target_folder = target_folder
        self.cmap = matplotlib.colors.ListedColormap(
            ["orange", "cyan", "green", "red", "blue"]
        )
        self.__plot_and_save()

    def __plot_and_save(self):
        for _, filename in enumerate(os.listdir(self.original_folder)):
            if filename.split(".")[-1] == "svs":
                print("image ", filename)

                slide = openslide.OpenSlide(self.original_folder + "/" + filename)
                image = slide.read_region(
                    (0, 0), LEVEL, slide.level_dimensions[LEVEL]
                ).convert("RGB")
                del slide

                downsampleFactor = 2

                new_x = math.floor(image.size[0]/downsampleFactor)
                new_y = math.floor(image.size[1]/downsampleFactor)
                image = image.resize((new_x, new_y), PIL.Image.BICUBIC)

                for _, source_filename in enumerate(os.listdir(self.source_folder)):
                    if source_filename.split("_")[0] == filename.split(".")[0]:
                        seg_matrix = np.load(self.source_folder + "/" + source_filename)
                        mask = (
                            np.arange(seg_matrix.max() + 1) == seg_matrix[..., None]
                        ).astype(bool)
                        fig = overlay_masks(
                            image, mask, mask_alpha=0.5, mpl_colormap=self.cmap
                        )
                        fig.savefig(
                            self.target_folder
                            + "/"
                            + source_filename.split(".")[0]
                            + ".png",
                            dpi=600,
                        )


if __name__ == "__main__":
    plotSegmentation(
        "/cluster/scratch/kkapusniak/WSS2-v1/test",
        "/cluster/scratch/kkapusniak/112_seg_map",
        "/cluster/scratch/kkapusniak/plots",
    )
