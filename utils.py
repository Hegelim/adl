"""Utility file"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray


INPUT_SIZE = 299
STRIDE = 128


def read_slide(slide, x, y, level, width, height, as_float=False):
    """Read a region from the slide, return a numpy RGB array
    
    See https://openslide.org/api/python/#openslide.OpenSlide.read_region
    Note: x,y coords are with respect to level 0.
    There is an example below of working with coordinates
    with respect to a higher zoom level.

    """
    im = slide.read_region((x,y), level, (width, height))
    im = im.convert('RGB') # drop the alpha channel
    if as_float:
        im = np.asarray(im, dtype=np.float32)
    else:
        im = np.asarray(im)
    assert im.shape == (height, width, 3)
    return im


def read_entire_slide(slide, level, as_float=False):
    """Read the entire slide at level"""
    return read_slide(
        slide, 0, 0, level, 
        slide.level_dimensions[level][0],
        slide.level_dimensions[level][1], as_float)



def find_tissue_pixels(image, intensity=0.8):
    """As mentioned in class, we can improve efficiency by ignoring non-tissue areas 
    of the slide. We'll find these by looking for all gray regions."""
    im_gray = rgb2gray(image)
    assert im_gray.shape == (image.shape[0], image.shape[1])
    indices = np.where(im_gray <= intensity)
    return list(zip(indices[0], indices[1]))


def apply_mask(im, mask, color=(255,0,0)):
    masked = np.copy(im)
    for x,y in mask: masked[x][y] = color
    return masked


def cal_tissue_perc(image, intensity=0.8) -> float:
    tissue_pixels = find_tissue_pixels(image)
    return len(tissue_pixels) / float(image.shape[0] * image.shape[0]) * 100


def extract_label(im, mask) -> int:
    """Extract label (0/1) from the center 128 x 128 of the im"""
    start_i = im.shape[0] // 2 - STRIDE // 2
    end_i = start_i + STRIDE
    start_j = im.shape[1] // 2 - STRIDE // 2
    end_j = start_j + STRIDE
    mask_slice = mask[start_i:end_i, start_j:end_j]
    return int(np.sum(mask_slice) >= 1)