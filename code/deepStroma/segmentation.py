from turtle import width
from PIL import Image, ImageOps
import numpy as np
from collections import defaultdict
import torch
from torchvision.models import resnet50, ResNet50_Weights

BATCH_SIZE = 8
CLASSIFIER_WIDTH = 224
CLASSIFIER_HEIGHT = 224


class Segmentation:
    def __init__(self, fun=None, padding=45, stride=90) -> None:
        if fun:
            self.__fun = fun
        else:
            self.__fun = self.__placeholder
        self.padding = padding
        self.stride = stride

    def get_probabilities(self, images) -> dict():
        self.__segment(images)

        probabilities = dict()
        for key, value in self.segmented_values.items():
            _, probabilities[key] = np.array(
                np.unique(value, return_counts=True)
            ) / len(value)
        return probabilities

    def __segment(self, images) -> None:
        for ind, image in enumerate(images):
            images[ind] = np.moveaxis(
                np.pad(
                    np.array(image),
                    [
                        (self.padding, self.padding),
                        (self.padding, self.padding),
                        (0, 0),
                    ],
                    mode="constant",
                ),
                -1,
                0,
            )

        (_, width, height) = images[0].shape
        width_n_steps = int(
            (width - CLASSIFIER_WIDTH + 2 * self.padding) / self.stride + 1
        )
        height_n_steps = int(
            (height - CLASSIFIER_HEIGHT + 2 * self.padding) / self.stride + 1
        )

        ## the buffer is to reduce number of calls to the classifier
        image_buffer = torch.empty((BATCH_SIZE, 3, CLASSIFIER_WIDTH, CLASSIFIER_HEIGHT))
        buffer_indices = list()
        self.segmented_values = defaultdict(list)
        for i, image in enumerate(images):
            for j in range(width_n_steps - 1):
                for k in range(height_n_steps - 1):
                    buffer_indices.append(i)

                    image_buffer[len(buffer_indices) - 1] = torch.tensor(
                        image[
                            :,
                            j * self.stride : j * self.stride + CLASSIFIER_WIDTH,
                            k * self.stride : k * self.stride + CLASSIFIER_HEIGHT,
                        ]
                    )

                    if len(buffer_indices) == BATCH_SIZE:
                        classes = self.__fun(image_buffer)
                        for ind in range(len(buffer_indices)):
                            self.segmented_values[buffer_indices[ind]].append(
                                classes[ind].detach().numpy()
                            )
                        image_buffer = torch.empty(
                            (BATCH_SIZE, 3, CLASSIFIER_WIDTH, CLASSIFIER_HEIGHT)
                        )
                        buffer_indices = list()

        if image_buffer != []:
            classes = self.__fun(image_buffer)
            for ind in range(len(buffer_indices)):
                self.segmented_values[buffer_indices[ind]].append(
                    classes[ind].detach().numpy()
                )

    ## to be replaced with our pytorch classifier
    def __placeholder(self, images) -> list():
        print("Batch -/-")
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        return model(images).squeeze(0).softmax(0)


if __name__ == "__main__":
    im1 = Image.open("TCGA-AA-3516.png")
    im2 = ImageOps.flip(Image.open("TCGA-AA-3516.png"))
    segment = Segmentation()
    print(segment.get_probabilities(np.array([im1, im2])))
