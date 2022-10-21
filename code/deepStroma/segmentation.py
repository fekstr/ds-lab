import math
from PIL import Image, ImageOps
import numpy as np
from collections import defaultdict
import torch
from torchvision.models import resnet50, ResNet50_Weights

BATCH_SIZE = 8
CLASSIFIER_WIDTH = 224
CLASSIFIER_HEIGHT = 224


class Segmentation:
    def __init__(
        self, fun=None, padding="keep_last_window", stride=CLASSIFIER_WIDTH
    ) -> None:
        if fun:
            self.__fun = fun
        else:
            self.__fun = self.__placeholder
        self.padding = padding
        self.stride = stride

    def get_segmentation_matrices(self, images) -> np.array:
        self.images = images
        self.__preprocess()
        self.__segment()

        self.segmentation_matrices = dict()
        for key, value in self.segmented_values.items():
            self.segmentation_matrices[key] = np.array(value).reshape(
                self.width_n_steps[key],
                self.height_n_steps[key],
            )

        self.__get_probabilities()

        return self.segmentation_matrices

    def __get_probabilities(self):
        probabilities = dict()
        for key, value in self.segmented_values.items():
            _, probabilities[key] = np.array(
                np.unique(value, return_counts=True)
            ) / len(value)
        self.probabilities = probabilities

    def __segment(self) -> None:
        ## the buffer is to reduce number of calls to the classifier
        image_buffer = torch.empty((BATCH_SIZE, 3, CLASSIFIER_WIDTH, CLASSIFIER_HEIGHT))
        buffer_indices = list()
        self.segmented_values = defaultdict(list)
        self.width_n_steps = [0] * len(self.images)
        self.height_n_steps = [0] * len(self.images)

        for i, image in enumerate(self.images):
            (_, width, height) = self.images[i].shape
            self.width_n_steps[i] = int(
                (width - CLASSIFIER_WIDTH + self.padding_width[i]) / self.stride
            )
            self.height_n_steps[i] = int(
                (height - CLASSIFIER_HEIGHT + self.padding_hight[i]) / self.stride
            )

            for j in range(self.width_n_steps[i]):
                for k in range(self.height_n_steps[i]):
                    buffer_indices.append(i)

                    image_buffer[len(buffer_indices) - 1] = torch.tensor(
                        image[
                            :,
                            j * self.stride : j * self.stride + CLASSIFIER_WIDTH,
                            k * self.stride : k * self.stride + CLASSIFIER_HEIGHT,
                        ]
                    )

                    if len(buffer_indices) == BATCH_SIZE:
                        print("Batch")
                        classes = self.__fun(image_buffer)
                        for ind in range(len(buffer_indices)):
                            self.segmented_values[buffer_indices[ind]].append(
                                np.argmax(classes[ind].detach().numpy())
                            )
                        image_buffer = torch.empty(
                            (BATCH_SIZE, 3, CLASSIFIER_WIDTH, CLASSIFIER_HEIGHT)
                        )
                        buffer_indices = list()

        if image_buffer != []:
            classes = self.__fun(image_buffer)
            for ind in range(len(buffer_indices)):
                self.segmented_values[buffer_indices[ind]].append(
                    np.argmax(classes[ind].detach().numpy())
                )

    def __preprocess(self):
        ## channel first
        for ind, image in enumerate(self.images):
            self.images[ind] = np.moveaxis(np.array(image), -1, 0)

        ## padding
        self.padding_width = list()
        self.padding_hight = list()

        for ind, image in enumerate(self.images):
            if self.padding == "keep_last_window":
                self.padding_width.append(image.shape[1] % CLASSIFIER_WIDTH)
                self.padding_hight.append(image.shape[2] % CLASSIFIER_WIDTH)
            else:
                self.padding_width.append(self.padding)
                self.padding_hight.append(self.padding)

        for ind, image in enumerate(self.images):
            np.pad(
                np.array(image),
                [
                    (0, 0),
                    (0, image.shape[1] % CLASSIFIER_WIDTH),
                    (0, image.shape[2] % CLASSIFIER_HEIGHT),
                ],
                mode="constant",
            )

    ## to be replaced with our pytorch classifier
    def __placeholder(self, images) -> list():
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        return model(images).squeeze(0).softmax(0)


if __name__ == "__main__":
    im1 = Image.open("TCGA-AA-3516.png")
    im2 = ImageOps.flip(Image.open("TCGA-AA-3516.png"))
    segment = Segmentation()
    print(segment.get_segmentation_matrices(np.array([im1, im2])))
