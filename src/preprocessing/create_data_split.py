import os
from pathlib import Path
from tqdm import tqdm

data_path = "data/PATH-DT-MSU-PROCESSED"
train_path = "data/PATH-DT-MSU-TRAIN"
test_path = "data/PATH-DT-MSU-TEST"


for cl in os.listdir(data_path):
    cl_path = os.path.join(data_path, cl)
    for file in tqdm(os.listdir(cl_path)):
        file_path = os.path.join(cl_path, file)
        if file.startswith("train"):
            dest_cl = os.path.join(train_path, cl)
        elif file.startswith("test"):
            dest_cl = os.path.join(test_path, cl)
        else:
            raise Exception("WTF")
        Path(dest_cl).mkdir(parents=True, exist_ok=True)
        dest = os.path.join(dest_cl, file)
        os.rename(file_path, dest)
