import os
import utils

def find_train_tif():
    masks = []
    slides = []
    for file in os.listdir(utils.train_input):
        if file.endswith(".tif"):
            filename = os.path.splitext(file)[0]
            if filename.endswith("mask"):
                masks.append(file)
            else:
                slides.append(file)
    masks.sort()
    slides.sort()
    slide_mask = list(zip(slides, masks))
    return slide_mask
    