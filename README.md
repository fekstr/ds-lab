# Data Science Lab

Formatter: [black](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter)  
Write [type hints](https://docs.python.org/3/library/typing.html)

Directory structure:

```
ds-lab
│   README.md
│
└───code
│   │   main.py
│   │   ...
│   │
│   └───models
│       │   resnet50.py
│       │   ...
│
└───data
    │
    └───CRC-VAL-HE-7K
     │   ...


```

## Report

[overleaf](https://www.overleaf.com/4317228738rvbdxvrjjsws)

## Relevant papers

[Tissue Type Recognition in Whole Slide Histological Images](http://ceur-ws.org/Vol-3027/paper50.pdf)

[Predicting survival from colorectal cancer histology slides using deep learning: A retrospective multicenter study](https://journals.plos.org/plosmedicine/article/file?id=10.1371/journal.pmed.1002730&type=printable)

## Data sets

[TCGA slides](https://portal.gdc.cancer.gov/repository?filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22content%22%3A%7B%22field%22%3A%22cases.case_id%22%2C%22value%22%3A%5B%22set_id%3ADqDl5YMBcZtnZdVfR3_w%22%5D%7D%2C%22op%22%3A%22IN%22%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.data_type%22%2C%22value%22%3A%5B%22Slide%20Image%22%5D%7D%7D%5D%7D)

[PATH-DT-MSU slides](https://imaging.cs.msu.ru/en/research/histology/path-dt-msu)

Download via command line

```
wget -O WSS2-v1.zip "https://downloader.disk.yandex.ru/disk/fe33dc2d46e32258d4d0c6aa8b7da62bbbddfad697dc81468ddc44ae7a5eebc9/638b95fd/fKqInKw3d7bLFOeFnMGnhAM2BMzL9eiNqnSmlFhPpUabc7-I5_ENJYdaJ64hLjyo2pjrh4ZbwUROGVvlVQPaGaLlbHiV4NwR6em5mcsuS5Kr8npumZHI4midPdWhecNq?uid=0&filename=WSS2-v1.zip&disposition=attachment&hash=k8z8FrQ/XqhbeBMqOVOEVSbAqNrbfsLWZ51dzBytdv0flGWevP2rJr7diMPOc87vW5aDQ4kMZEXE%2BwNjbq78ug%3D%3D%3A&limit=0&content_type=application%2Fzip&owner_uid=1130000056703611&fsize=20938883225&hid=90683e66eeabc6626744a7c43937c80c&media_type=compressed&tknv=v2"  -q --show-progress
```

[Processed TCGA slides](https://polybox.ethz.ch/index.php/s/g9pXo8JraNZIdNx)

[PATH slides to patches](https://polybox.ethz.ch/index.php/s/eAGHwpMehXX0Rrg)

### Normalized patches

[PATH-DT-MSU-TRAIN](https://polybox.ethz.ch/index.php/s/hzLxnW3Lb64V7UA/download)

[PATH-DT-MSU-TEST](https://polybox.ethz.ch/index.php/s/GOl8C9ONgDdxvBr/download)

## Trained models

[EfficientNet B0](https://polybox.ethz.ch/index.php/s/asyU2O8VKnxi6gd/download)

[EfficientNet B0 (tuned on PATH dataset)](https://polybox.ethz.ch/index.php/s/nX0SVRoob5X5uR0)

## Segmentation Results

#### Numpy matrices for PATH test

[Non-overlapping (stride=224)](https://polybox.ethz.ch/index.php/s/m3GHWyVELXprBt5)

[Overlapping (stride=112)](https://polybox.ethz.ch/index.php/s/WYFPf3kv24bhx8W)

[Overlapping (stride=28)](https://polybox.ethz.ch/index.php/s/mVQA4SZcGgWRqaB)

#### Segmentation plots for PATH test

[All Images for report](https://polybox.ethz.ch/index.php/s/sKvRP7UBLbdl6uI)

#### CSV files for DEEP STROMA score - eqivalent of their excel files

[TCGA patients with probabilities averaged](https://polybox.ethz.ch/index.php/s/F800Obkts8TEE6Y)

[TCGA patients with probabilities - most probable tumor image](https://polybox.ethz.ch/index.php/s/8SlUq2AFCZC0bSn)

#### Input for postprocess

[TCGA: probabilities per image - unfiltered](https://polybox.ethz.ch/index.php/s/f7Ofp4ZcShN1lmk)
