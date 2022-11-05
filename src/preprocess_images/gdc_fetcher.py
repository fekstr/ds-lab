import urllib
import urllib.request
import os
import PIL
from PIL import Image
from preprocess_image import PreprocessingSVS
import sys
from pathlib import Path

batchNo = int(sys.argv[1]) if len(sys.argv) > 1 else -1
outputFolder = sys.argv[2]
baseUrl = 'https://api.gdc.cancer.gov'

batchSize = 100

with open('gdc_manifest.2022-10-18.txt') as f:
    lines = f.readlines()[1:]
    n = len(lines)
    indeces = range(min(n, batchSize * batchNo), min(n, batchSize * batchNo + batchSize)) if batchNo != -1 else range(n)
    for i in indeces:
        line = lines[i]
        id, name = line.split('\t')[:2]
        nameParts = name.split('.')
        name = f'{nameParts[0]}.{nameParts[2]}'
        folderPath = f'{outputFolder}/TCGA_processed'
        path = f'{folderPath}/{name}'
        try: 
            Path(folderPath).mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f'{baseUrl}/data/{id}', path)
            preprocess = PreprocessingSVS(path)
            preprocess.resize_to_target_mpp()
            preprocess.crop()
            preprocess.normalise()
            preprocess.save()
            os.remove(path)
        except Exception as e:
            print(f'Error in processing {name}. Error: {e}')
            with open(f'{outputFolder}/failed_tcga.txt', 'a') as fd:
                fd.write(f'{id} \t {name} \n')
        # break # Add break so if you accidentally run this you won't dowload the entire dataset. Adjust this code as needed
