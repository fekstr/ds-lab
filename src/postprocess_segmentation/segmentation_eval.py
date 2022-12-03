OPENSLIDE_PATH = r'C:\\Users\\dantg\\openslide-win64-20220811\\openslide-win64-20220811\\bin'

import cv2
import json
import os
import math
import PIL
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

classLabels = {
    'AT': 1,
    'BG': 2,
    'LP': 3,
    'MM': 4,
    'TUM': 5,
}

def calcualteIouPerClass(maskTrue, maskPred):
    iou = {}
    for label, value in classLabels.items():
        maskPredLabel = maskPred == value
        maskTrueLabel = maskTrue == value
        maskPredLabel = np.multiply(maskPredLabel, maskTrue > 0)
        iou[label] = iouScore(maskTrueLabel, maskPredLabel)

    return iou

def iouScore(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    unionSum = np.sum(union)
    iou_score = np.sum(intersection) / unionSum if unionSum > 0 else 0
    return iou_score

groundTruthPath = sys.argv[1]
predictedPath = sys.argv[2]
outputFolder = sys.argv[3]

print("Calculating IoU for ", groundTruthPath, " and ", predictedPath)

groundTruh = np.load(groundTruthPath)
predicted = np.load(predictedPath)

print("Ground truth shape: ", groundTruh.shape)
print("Predicted shape: ", predicted.shape)
iou = calcualteIouPerClass(groundTruh, predicted)

print("IOU ", iou)


