## Introduction

### Prerequisites

1. Disk size
2. GPU
3. tensorflow environment
4. Memory

### How to Reproduce
<!-- 1. run `prepare_data.ipynb` on Google Colab -->

1. download tumor slide .tif images and mask .tif images to local
drive
2. run `python create_training_slices.py`
3. run `train.py` to train the model


### Training Data Preparation

* 1000 slices each from tumor_012, tumor_091, tumor_031, tumor_110, tumor_035, tumor_057, tumor_059, tumor_064

### Test Data Preparation

