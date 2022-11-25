import os
import math
from PIL import Image
import openslide
import numpy as np
import pandas as pd
from collections import defaultdict
import torch
from torchvision.transforms import Normalize, ToTensor
from torchvision import transforms
import torchstain
import PIL

from models.pretrained_classification_model import ImgClassificationModel

BATCH_SIZE = 32
CLASSIFIER_WIDTH = 224
CLASSIFIER_HEIGHT = 224
NUM_CLASSES = 5


class Segmentation:
    def __init__(
        self,
        fun_checkpoint=None,
        padding="keep_last_window",
        stride=CLASSIFIER_WIDTH,
        normalise=False,
    ) -> None:
        """Initialise segmentation

        Args:
            fun (_type_, optional): Classifier function. Defaults to None.
            padding (str, optional): By default adds padding so that segmentation map is the same size as image for stride = CLASSIFIER_WIDTH. Defaults to "keep_last_window".
            stride (_type_, optional): By default non-overlapping, should be at most CLASSIFIER_WIDTH. Defaults to CLASSIFIER_WIDTH.
        """

        self.__fun = self.__pytorch_model
        self.model = ImgClassificationModel.load_from_checkpoint(
            fun_checkpoint, num_classes=NUM_CLASSES
        )
        self.model.freeze()

        self.padding = padding
        self.stride = stride

        self.macenko_normalise = normalise

        if self.macenko_normalise:
            self.T = transforms.Compose(
                [transforms.ToTensor(), transforms.Lambda(lambda x: x * 255)]
            )
            self.torch_normaliser = torchstain.normalizers.MacenkoNormalizer(
                backend="torch"
            )
            target = PIL.Image.open("/cluster/scratch/kkapusniak/Ref.png")
            self.torch_normaliser.fit(self.T(target))

    def create_TCGA_spreadsheet(
        self, folder_location: str, buffer_frequency=20, checkpoint_frequency=100
    ):
        results = dict()
        image_buffer = []

        for ind, filename in enumerate(os.listdir(folder_location)):
            if filename.split(".")[-1] == "tif":
                print("image ", ind)

                image_buffer.append(filename)

                if len(image_buffer) == buffer_frequency:
                    self.images = [
                        Image.open(folder_location + "/" + fn) for fn in image_buffer
                    ]
                    self.__probabilities_only_sequence()
                    for key, value in self.probabilities.items():
                        results[image_buffer[key]] = value
                    image_buffer = []

                if ind % checkpoint_frequency == 0:
                    pd.DataFrame(results).T.to_csv(
                        "TCGA_probabilities_per_image.csv", float_format="%.18f"
                    )

        if len(image_buffer) != 0:
            self.images = [
                Image.open(folder_location + "/" + fn) for fn in image_buffer
            ]
            self.__probabilities_only_sequence()
            for key, value in self.probabilities.items():
                results[image_buffer[key]] = value

        pd.DataFrame(results).T.to_csv(
            "TCGA_probabilities_per_image.csv", float_format="%.18f"
        )

    def segment_PATH(self, folder_location: str, save_location: str):
        for ind, filename in enumerate(os.listdir(folder_location)):

            if filename.split(".")[-1] == "svs":
                print("image ", ind)

                slide = openslide.OpenSlide(folder_location + "/" + filename)

                self.images = [
                    slide.read_region((0, 0), 1, slide.level_dimensions[1]).convert(
                        "RGB"
                    )
                ]
                del slide
                self.__segmentation_only_sequence()

                with open(
                    save_location
                    + "/"
                    + filename.split(".")[0]
                    + "_"
                    + str(self.stride)
                    + "_segmentation_map.npy",
                    "wb",
                ) as f:
                    np.save(f, self.segmentation_matrices[0])

    def __probabilities_only_sequence(self):
        self.__preprocess()
        self.__segment()
        self.__get_probabilities()

    def __segmentation_only_sequence(self):
        self.__preprocess()
        self.__segment(normalise=True)
        self.__assemble_segments()

    def __preprocess(self) -> None:
        """SET Channel first and add padding for each image"""
        for ind, image in enumerate(self.images):
            self.images[ind] = np.array(image)

        self.padding_width = list()
        self.padding_hight = list()

        for ind, image in enumerate(self.images):
            if self.padding == "keep_last_window":
                self.padding_width.append(
                    math.ceil(
                        (CLASSIFIER_WIDTH - image.shape[0] % CLASSIFIER_WIDTH) / 2
                    )
                )
                self.padding_hight.append(
                    math.ceil(
                        (CLASSIFIER_WIDTH - image.shape[1] % CLASSIFIER_WIDTH) / 2
                    )
                )
            else:
                self.padding_width.append(self.padding)
                self.padding_hight.append(self.padding)

        for ind, image in enumerate(self.images):
            self.images[ind] = np.pad(
                np.array(image),
                [
                    (self.padding_width[ind], self.padding_width[ind]),
                    (self.padding_hight[ind], self.padding_hight[ind]),
                    (0, 0),
                ],
                mode="constant",
            )

    def __segment(self, normalise=False) -> None:
        """Segments images into segments and pass them to classifier, grouped into batches of BATCH_SIZE"""

        image_buffer = np.zeros(
            (BATCH_SIZE, CLASSIFIER_WIDTH, CLASSIFIER_HEIGHT, 3), dtype=np.uint8
        )
        buffer_indices = list()
        self.segmented_values = defaultdict(list)
        self.width_n_steps = [0] * len(self.images)
        self.height_n_steps = [0] * len(self.images)

        batch_num = 0

        for i, image in enumerate(self.images):
            (width, height, _) = self.images[i].shape
            self.width_n_steps[i] = int((width + self.padding_width[i]) / self.stride)
            self.height_n_steps[i] = int((height + self.padding_hight[i]) / self.stride)
            approx_num_of_batches = (
                len(self.images)
                * (self.width_n_steps[i] * self.height_n_steps[i])
                // BATCH_SIZE
            )

            for j in range(self.width_n_steps[i]):
                for k in range(self.height_n_steps[i]):
                    buffer_indices.append(i)

                    image_buffer[len(buffer_indices) - 1] = image[
                        j * self.stride : j * self.stride + CLASSIFIER_WIDTH,
                        k * self.stride : k * self.stride + CLASSIFIER_HEIGHT,
                        :,
                    ]

                    if len(buffer_indices) == BATCH_SIZE:
                        print(
                            "Batch ",
                            batch_num,
                            "/",
                            approx_num_of_batches,
                            "(approx if varying image size)",
                        )
                        batch_num += 1

                        classes = self.__fun(image_buffer)
                        for ind in range(len(buffer_indices)):
                            self.segmented_values[buffer_indices[ind]].append(
                                classes[ind].detach().numpy()
                            )
                        image_buffer = np.zeros(
                            (BATCH_SIZE, CLASSIFIER_WIDTH, CLASSIFIER_HEIGHT, 3),
                            dtype=np.uint8,
                        )
                        buffer_indices = list()

        if image_buffer != []:
            print("Last Batch")
            classes = self.__fun(image_buffer)
            for ind in range(len(buffer_indices)):
                self.segmented_values[buffer_indices[ind]].append(
                    classes[ind].detach().numpy()
                )

    def __assemble_segments(self) -> None:
        """generates matrices, with resolutions corresponding to each picture, with most probable class at each pixel"""

        self.segmentation_matrices = dict()
        probabilities = dict()

        ## supports overlapping segments (sums up probabilities of each pixel)
        for key, value in self.segmented_values.items():
            print("Generating segmentation map for image", key)

            probabilities = np.zeros(
                (self.images[key].shape[0], self.images[key].shape[1], NUM_CLASSES)
            )
            for j in range(self.width_n_steps[key] + 1):
                for k in range(self.height_n_steps[key] + 1):
                    probabilities[
                        j * self.stride : j * self.stride + CLASSIFIER_WIDTH,
                        k * self.stride : k * self.stride + CLASSIFIER_HEIGHT,
                    ] += value[j + k]
            self.segmentation_matrices[key] = np.argmax(probabilities, axis=2)

    def __get_probabilities(self) -> None:
        """probabilities for deep stroma score as in original paper"""

        probabilities = dict()
        for key, value in self.segmented_values.items():
            probabilities[key] = np.mean(value, axis=0)
        self.probabilities = probabilities

    def __pytorch_model(self, images: np.array) -> list():
        normaliser = Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True
        )

        test_dataset = torch.zeros(
            images.shape[0], images.shape[3], images.shape[1], images.shape[2]
        )
        for ind, image in enumerate(images):
            if self.macenko_normalise:
                try:
                    norm, _, _ = self.torch_normaliser.normalize(
                        I=self.T(image.astype(np.uint8)), stains=False
                    )
                    image = np.uint8(norm.numpy())
                except:
                    print("WARNING: Normalisation skipped")

            test_dataset[ind] = ToTensor()(image.astype(np.uint8))
        normaliser(test_dataset)

        with torch.no_grad():
            pred = self.model.model(test_dataset).softmax(1)
        return pred


if __name__ == "__main__":
    if NUM_CLASSES == 9:
        segment = Segmentation(
            fun_checkpoint="version_1884922/checkpoints/epoch=9-step=3130.ckpt",
        )
        segment.create_TCGA_spreadsheet(folder_location="./TCGA_processed")

    if NUM_CLASSES == 5:
        segment = Segmentation(
            fun_checkpoint="/cluster/scratch/kkapusniak/version_2648177/checkpoints/last.ckpt",
            normalise=True,
        )
        segment.segment_PATH(
            folder_location="/cluster/scratch/kkapusniak/WSS2-v1/test",
            save_location="/cluster/scratch/kkapusniak/",
        )
