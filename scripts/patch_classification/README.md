# Patch Classification

In the patch classification task, the goal is to predict a tissue class from an image patch. The input is a normalized tensor of shape (3, 224, 224) and the output is one of <i>k</i> classes.

## Datasets

We train and evaluate the classifier on two different datasets. Detailed descriptions of the datasets are available at the respective links. Here, we provide short summaries.

<b>[NCT-CRC-HE-100K](https://zenodo.org/record/1214456#.Y4ihluzMI-Q)</b>

This is a dataset of 100K image patches from H&E-stained histological images of human colorectal cancer and normal tissue. Each patch is labeled with one of 9 tissue classes.

<b>[PATH-DT-MSU](https://imaging.cs.msu.ru/en/research/histology/path-dt-msu)</b>

This is a dataset of whole-slide H&E-stained histological images of human colon- cancer, stomach cancer, and normal tissue. The slides are annotated with polygons indicating 5 tissue classes.

## Method

We train and evaluate several state-of-the-art classifiers on the task. The included classifiers are ResNet50, VGG19 and EfficientNet B0.

We hypothesize that pretraining on one of the datasets followed by tuning on the second dataset will yield better performance than training the classifier on the second dataset directly.

## Evaluation

Hold-out test sets are supplied together with the training data at the dataset links. We evaluate the classification models on these test sets.

The impact on classification performance is evaluated for the following

- Model choice
- Pre-training versus training from scratch
