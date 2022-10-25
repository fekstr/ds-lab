import urllib
import urllib.request
import os
import PIL
from PIL import Image
from preprocess_image import PreprocessingSVS
baseUrl = 'https://api.gdc.cancer.gov'

with open('gdc_manifest.2022-10-18.txt') as f:
    for line in f.readlines()[1:]:
        id, name = line.split('\t')[:2]
        nameParts = name.split('.')
        name = f'{nameParts[0]}.{nameParts[2]}'
        path = f'slides/{name}'
        urllib.request.urlretrieve(f'{baseUrl}/data/{id}', path)
        preprocess = PreprocessingSVS(path)
        preprocess.resize_to_target_mpp()
        preprocess.crop()
        preprocess.normalise()
        preprocess.save()
        os.remove(path)
        break # Add break so if you accidentally run this you won't dowload the entire dataset. Adjust this code as needed
