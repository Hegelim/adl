"""Utility file"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray


INPUT_SIZE = 299
TRAIN_STRIDE = 30
TEST_STRIDE = 128

TRAIN_SAMPLE_SIZE = 1000

# define input paths
train_input = "./TIFs/trainingTIFs"
train_input_mask = "./TIFs/testingTIFs"

# define zoom levels
# assume level1 < level2
# which means image at level 1 has 
# more details than level 2
level1 = 3
level2 = 4

training_dir = "./expTraining"
zoom1_patches = "./expTraining/zoom1/slides"
zoom2_patches = "./expTraining/zoom2/slides"

# define training output paths
zoom1_tumor_patches = f"{training_dir}/zoom1/slides/tumor"
zoom1_normal_patches = f"{training_dir}/zoom1/slides/normal"
zoom1_tumor_masks = f"{training_dir}/zoom1/masks/tumor"
zoom1_normal_masks = f"{training_dir}/zoom1/masks/normal"

zoom2_tumor_patches = f"{training_dir}/zoom2/slides/tumor"
zoom2_normal_patches = f"{training_dir}/zoom2/slides/normal"
zoom2_tumor_masks = f"{training_dir}/zoom2/masks/tumor"
zoom2_normal_masks = f"{training_dir}/zoom2/masks/normal"

# seed for ImageDataGenerator
seed = 100

train_batch_size = 32

intensity = 0.8

def read_slide(slide, x, y, level, width, height, as_float=False):
    """Read a region from the slide, return a numpy RGB array
    
    See https://openslide.org/api/python/#openslide.OpenSlide.read_region
    Note: x,y coords are with respect to level 0.
    There is an example below of working with coordinates
    with respect to a higher zoom level.
    here x,y is w.r.t. images
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


def cal_tissue_perc(image) -> float:
    tissue_pixels = find_tissue_pixels(image)
    return len(tissue_pixels) / (image.shape[0] * image.shape[1])


def has_tumor_at_center(mask) -> bool:
    """Extract label (0/1) from the center 128 x 128 of the im
    
    Args:
        mask: 2d array, (299, 299)
    """
    start_i = mask.shape[0] // 2 - TEST_STRIDE // 2
    end_i = start_i + TEST_STRIDE
    start_j = mask.shape[1] // 2 - TEST_STRIDE // 2
    end_j = start_j + TEST_STRIDE
    mask_slice = mask[start_i:end_i, start_j:end_j]
    return np.count_nonzero(mask_slice) >= 1