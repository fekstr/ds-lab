from tkinter import Y
import openslide

image = openslide.OpenSlide("TCGA-AA-3516.svs")

print(image.properties)
print(image.properties['openslide.mpp-x'])
print(image.properties['openslide.mpp-y'])