from PIL import Image, ImageOps
import numpy as np
from collections import defaultdict
import torch
from torchvision.models import resnet50, ResNet50_Weights

BATCH_SIZE = 8
CLASSIFIER_WIDTH = 224
CLASSIFIER_HEIGHT = 224
NUM_CLASSES = 1000


class Segmentation:
    def __init__(
        self, fun=None, padding="keep_last_window", stride=CLASSIFIER_WIDTH
    ) -> None:
        """Initialise segmentation

        Args:
            fun (_type_, optional): Classifier function. Defaults to None.
            padding (str, optional): By default adds padding so that segmentation map is the same size as image for stride = CLASSIFIER_WIDTH. Defaults to "keep_last_window".
            stride (_type_, optional): By default non-overlapping, should be at most CLASSIFIER_WIDTH. Defaults to CLASSIFIER_WIDTH.
        """
        if fun:
            self.__fun = fun
        else:
            self.__fun = self.__placeholder
        self.padding = padding
        self.stride = stride

    def get_segmentation_matrices(self, images) -> dict():
        """Calls all function in order

        Args:
            images (_type_): np.array of full sized images

        Returns:
            np.array: dictonary of segmentation maps (np.arrays), key corresponds to image ID
        """
        self.images = images

        self.__preprocess()
        self.__segment()
        self.__assemble_segments()
        self.__get_probabilities()

        return self.segmentation_matrices

    def __preprocess(self) -> None:
        """SET Channel first and add padding for each image"""

        for ind, image in enumerate(self.images):
            self.images[ind] = np.moveaxis(np.array(image), -1, 0)

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
                    (0, self.padding_width[ind]),
                    (0, self.padding_hight[ind]),
                ],
                mode="constant",
            )

    def __segment(self) -> None:
        """Segments images into segments and pass them to classifier, grouped into batches of BATCH_SIZE"""

        image_buffer = torch.empty((BATCH_SIZE, 3, CLASSIFIER_WIDTH, CLASSIFIER_HEIGHT))
        buffer_indices = list()
        self.segmented_values = defaultdict(list)
        self.width_n_steps = [0] * len(self.images)
        self.height_n_steps = [0] * len(self.images)

        batch_num = 0

        for i, image in enumerate(self.images):
            (_, width, height) = self.images[i].shape
            self.width_n_steps[i] = int(
                (width - CLASSIFIER_WIDTH + self.padding_width[i]) / self.stride
            )
            self.height_n_steps[i] = int(
                (height - CLASSIFIER_HEIGHT + self.padding_hight[i]) / self.stride
            )

            approx_num_of_batches = (
                len(self.images)
                * (self.width_n_steps[i] * self.height_n_steps[i])
                // BATCH_SIZE
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
                        image_buffer = torch.empty(
                            (BATCH_SIZE, 3, CLASSIFIER_WIDTH, CLASSIFIER_HEIGHT)
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
                (self.images[key].shape[1], self.images[key].shape[2], NUM_CLASSES)
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
            _, probabilities[key] = np.array(
                np.unique(value, return_counts=True)
            ) / len(value)
        self.probabilities = probabilities

    ## to be replaced with our pytorch classifier
    def __placeholder(self, images) -> list():
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        return model(images).squeeze(0).softmax(0)


if __name__ == "__main__":
    im1 = Image.open("preprocess_full_images/TCGA-AA-3516.png")
    im2 = ImageOps.flip(Image.open("preprocess_full_images/TCGA-AA-3516.png"))
    segment = Segmentation()
    print(segment.get_segmentation_matrices(np.array([im1, im2])))
