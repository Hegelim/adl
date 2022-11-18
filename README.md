## Introduction

This project aims to create a deep learning end-to-end pipeline that helps to identify and localize cancer cells in gigapixel pathology images

* Research Paper: https://arxiv.org/abs/1703.02442
* Data: https://camelyon16.grand-challenge.org/Data/
* Youtube: 

## Environment

The repo is based on local development using WSL2 with a Nvidia GPU. Below is a set of hardware usage based on current repo setting. Because the project involves handling big size of data, a more capable set of hardware or using Cloud services is recommended

* Disk size: 31GB
* GPU: Nvidia GeForce 3060 Laptop
* Memory: 32GB

Software environment:

* python 3.9
* tensorflow with conda
* WSL 2

The virtual environment can be created using 
```
conda create --name <env> --file requirements.txt
```

## Ideas in a Nutshell

### Models

* transfer learning in InceptionV3
* customized CNN model

The checkpoints are so large that it is unfeasible to store in github. Instead, they can be downloaded with the links

* [InceptionV3_small_30](https://drive.google.com/file/d/1zFgo7c1HrH9spK88pqwYjwRs4PGi7-SX/view?usp=share_link)



### Input

* Size: (299, 299, 3)

### Training Data
```
>>> import create_training_dataset
>>> create_training_dataset.create_train_val_dataset()
Found 11646 images belonging to 2 classes.
Found 2911 images belonging to 2 classes.
Found 11646 images belonging to 2 classes.
Found 2911 images belonging to 2 classes.
{'normal': 0, 'tumor': 1}
``` 

### Training Benchmark

Using model `checkpoints/inceptionv3_small_11_17_categorical_30.h5`
![image](plots/Accuracy_inceptionv3_small_11_17_categorical_30.png)

![image](plots/Auc_inceptionv3_small_11_17_categorical_30.png)

![image](plots/Recall_inceptionv3_small_11_17_categorical_30.png)

### Testing

I used tumor_078 because it has a relatively large region of tumor. Using model `inceptionv3_small_11_17_categorical_30.h5`
![image](plots/inceptionv3_small_11_17_categorical_30_step1_tumor_078_level7.png)


## Structure

The structure of the repo looks like follows
```
❯ tree -d
.
├── TIFs
│   ├── testingTIFs
│   └── trainingTIFs
├── checkpoints
├── experiment
├── history
├── notebooks
├── plots
├── predictions
└── training
    ├── zoom1
    │   ├── masks
    │   │   ├── normal
    │   │   └── tumor
    │   └── patches
    │       ├── normal
    │       └── tumor
    └── zoom2
        ├── masks
        │   ├── normal
        │   └── tumor
        └── patches
            ├── normal
            └── tumor
```
In above structure
* TIFs: store original TIF images
* notebooks: store jupyter notebooks
* training: store training patches
* checkpoints: store model checkpoints
* history: compressed model `history.history` in `.pkl` format
* plots: figures
* predictions: prediction lists in `.pkl` format

## How to Reproduce
<!-- 1. run `prepare_data.ipynb` on Google Colab -->
It is recommened to skim through all the python scripts and be familiar with the general structure. Specifically, it is important to understand and make sure all global variables in `utils.py` are correctly configured

### Download Data to Local

download tumor slide .tif images and mask .tif images to local
drive. By default, split training slide and tumor tif images to `./TIFs/trainingTIFs/` and testing slide and tumor images to `./TIFs/testingTIFs/`

### Create Training Patches

Because the original image is very large (~1GB per slide), we will follow the paper and create smaller training patches from original images. To accomodate the usage of InceptionV3, we will make (299, 299) patches of data from a certain level of the image

First, make sure all global variables in `utils.py` are set correctly, then

```
❯ python create_training_slices.py
```

In our case, we used images tumor_012, tumor_091, tumor_031, tumor_110, tumor_035, tumor_057, tumor_059, tumor_064

The total number of normal patches (per each zoom level) is

```
❯ ls training/zoom1/patches/normal/ | wc -l
8000
```

The total number of tumor patches (per each zoom level) is

```
❯ ls training/zoom1/patches/tumor/ | wc -l
6557
```

### (Optional) Validate Training Slices

To validate we have created normal/tumor patches correctly on 2 different zoom levels (which is complicated!), initiate jupyter
```
❯ jupyter lab
```
Then open and run `./validate_training_data.ipynb`

### Training 

If needed, modify or add model in `model.py`, then 
specify output path to save checkpoints in `train.py`, then

```
❯ python train.py
```

Which will read generator created by `create_training_dataset.py`

### Plot Heatmap and Evaluate

Initiate jupyter lab session by
```
❯ jupyter lab
```

Then open and run `./notebooks/make_prediction.ipynb`
the plots are saved in `./plots/`

### Plot History

Initiate jupyter lab session by
```
❯ jupyter lab
```

Then open and run `./notebooks/plot_history.ipynb`