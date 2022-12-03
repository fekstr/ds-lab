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
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

type = sys.argv[1]
index = int(sys.argv[2])
datapath = sys.argv[3]
outputFolder = sys.argv[4]

classLabels = {
    'AT': 1,
    'BG': 2,
    'LP': 3,
    'MM': 4,
    'TUM': 5,
}

json_filename = os.path.join(datapath, type, f'0{index}_anno.json')
json_file = open(json_filename, "r")
annos = json.load(json_file)
print('opening slide')
slideName = os.path.join(datapath, type, f'0{index}.svs')
slide = openslide.OpenSlide(slideName)
print('Reading image')
level = 1
image = slide.read_region(
                    (0, 0), level, slide.level_dimensions[level]
                )
del slide
print('Done reading image')
downsampleFactor = 4

new_x = math.floor(image.size[0] / downsampleFactor)
new_y = math.floor(image.size[1] / downsampleFactor)

# image = image.resize((new_x, new_y), PIL.Image.BICUBIC)

mask = np.zeros((image.size[1], image.size[0]), dtype=np.int32)
# mask_pred = np.zeros((image.size[1], image.size[0]), dtype=np.int32)

# shape = slide.level_dimensions[0]
# mask = np.zeros((shape[1], shape[0]), dtype=np.int32)
# mask_pred = np.zeros((shape[1], shape[0]), dtype=np.int32)


# print(mask.shape)
for poly in annos:
    cv2.fillPoly(mask, [np.array(poly['vertices'],dtype=np.int32) // downsampleFactor ], classLabels[poly['class']])

# iou = calcualteIouPerClass(mask, mask_pred)
# print('iou', iou)


folder = f'{outputFolder}/masks'
Path(f'{outputFolder}/masks').mkdir(parents=True, exist_ok=True)

np.save(os.path.join(folder, f'{type}_{index}_mask.npy'), mask)

# print('Showing image')
# f, axarr = plt.subplots(2,2)
# axarr[0, 0].imshow(image)
# axarr[0, 1].imshow(mask)
# axarr[1, 0].imshow(image)
# axarr[1, 1].imshow(mask_pred)
# plt.show()




