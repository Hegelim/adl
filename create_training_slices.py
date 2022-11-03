"""Create training dataset

Usage:
    python create_training_dataset.py

Author: Yewen Zhou
Date: 11/3/2022
"""
import utils
import numpy as np
from openslide import open_slide
import os
import cv2
from shutil import rmtree


def extract_patch_from_center(
    slide, mask,
    center_x, center_y, 
    level1, level2, 
    patch_size):
    """Extract 2 patches from same slide at 2 zoom levels 
    centered at same region
    Assert:
        level1 < level2: level1 has smaller downsample rate, level1 for zoomed-in version
    
    Args:
        slide: open_slide object
        level1, level2: int, e.g. 4, 5
        center_x, center_y: coordinates of center w.r.t. level1 (not downsample 1)
        patch_size: size of the patch w.r.t. level1, int, eg 299, should be same for all levels

    Note:
        the function itself doesn't check whether the image is out of bound
        that is left for other validation functions

    Returns:
        a dictionary that has 4 k-v pairs
    """
    assert level1 < level2 
    level1_downsample_fac = int(slide.level_downsamples[level1])
    level2_downsample_fac = int(slide.level_downsamples[level2])
    
    start_x_level1 = center_x - patch_size // 2
    start_y_level1 = center_y - patch_size // 2

    # the starting point of level1 w.r.t. level 0
    start_x_level10 = start_x_level1 * level1_downsample_fac
    start_y_level10 = start_y_level1 * level1_downsample_fac

    # =============================
    # =starting point leve2->0=========
    center_x_level2 = center_x * level1_downsample_fac // level2_downsample_fac
    center_y_level2 = center_y * level1_downsample_fac // level2_downsample_fac

    start_x_level2 = center_x_level2 - patch_size // 2
    start_y_level2 = center_y_level2 - patch_size // 2

    # the starting point of level2 w.r.t. level 0
    start_x_level20 = start_x_level2 * level2_downsample_fac
    start_y_level20 = start_y_level2 * level2_downsample_fac

    level1_patch = utils.read_slide(slide, start_x_level10, start_y_level10, level1, patch_size, patch_size)
    level2_patch = utils.read_slide(slide, start_x_level20, start_y_level20, level2, patch_size, patch_size)
    
    level1_mask = utils.read_slide(mask, start_x_level10, start_y_level10, level1, patch_size, patch_size)
    level2_mask = utils.read_slide(mask, start_x_level20, start_y_level20, level2, patch_size, patch_size)
    
    res = {}
    res['level1_patch'] = level1_patch
    res['level2_patch'] = level2_patch
    res['level1_mask'] = level1_mask
    res['level2_mask'] = level2_mask
    
    return res


def validate_image(image, threshold=0.4) -> bool:
    """Validate the image

    Args:
        image: an RGB image    
        
    Returns:
        True if image is valid
    """
    return utils.cal_tissue_perc(image) >= threshold


def extract_patches_to_dir(
    slide_path, mask_path, level1, level2, stride, input_size, 
    zoom1_patch_outdir, zoom1_label_outdir, zoom1_mask_outdir, 
    zoom2_patch_outdir, zoom2_label_outdir, zoom2_mask_outdir) -> None:
    """Extract patches from im to dir
    1. Assume we will extract patches from the entire image at 2 zoom levels
    2. 

    """
    # level1 - zoomed-in version
    # level2 - zoomed-out version
    assert level1 < level2
    
    slide = open_slide(slide_path)
    mask = open_slide(mask_path)
    slide_name = os.path.splitext(slide_path)[0]
    # we perform operations w.r.t. level1
    # the zoomed-in level
    r, c = slide.level_dimensions[level1]

    i, count = 0, 1
    level1_labels, level2_labels = [], []
    while i * stride + input_size < r:
        j = 0
        while j * stride + input_size < c:
            end_i, end_j = i * stride + input_size, j * stride + input_size
            center_x, center_y = (j + end_j) // 2, (i + end_i) // 2
            
            patch_mask = extract_patch_from_center(slide, mask, center_x, center_y, level1, level2, input_size)
            patch_level1, patch_level2 = patch_mask['level1_patch'], patch_mask['level2_patch']
            mask_level1, mask_level2 = patch_mask['level1_mask'], patch_mask['level2_mask']
            
            if validate_image(patch_level1) and validate_image(patch_level2):
                cv2.imwrite(f"{zoom1_patch_outdir}/{slide_name}_{count}.png", patch_level1)
                cv2.imwrite(f"{zoom2_patch_outdir}/{slide_name}_{count}.png", patch_level2)
                cv2.imwrite(f"{zoom1_mask_outdir}/{slide_name}_mask_{count}.png", mask_level1)
                cv2.imwrite(f"{zoom2_mask_outdir}/{slide_name}_mask_{count}.png", mask_level2)
                level1_labels.append(utils.extract_label(patch_level1, mask_level1))
                level2_labels.append(utils.extract_label(patch_level2, mask_level2))
                count += 1
            j += 1
        i += 1
        
    with open(f"{zoom1_label_outdir}/zoom1labels.npy", 'wb') as f:
        np.save(f, np.array(level1_labels))
    
    with open(f"{zoom2_label_outdir}/zoom2labels.npy", 'wb') as f:
        np.save(f, np.array(level2_labels))
        
    print("finished!")
    

def make_dirs(dirs, rerun=False):
    for dir in dirs:
        if os.path.isdir(dir):
            if rerun:
                rmtree(dir)
                os.makedirs(dir)
        else:
            os.makedirs(dir)


if __name__ == "__main__":
    dirs = [
        utils.zoom1_patches,
        utils.zoom1_labels,
        utils.zoom1_masks,
        utils.zoom2_patches,
        utils.zoom2_labels,
        utils.zoom2_masks,
    ]
    
    # set rerun=True to reset directories
    make_dirs(dirs, rerun=True)

    extract_patches_to_dir(
        utils.train_input_im, utils.train_input_mask, utils.level1, utils.level2, utils.STRIDE, utils.INPUT_SIZE, 
        utils.zoom1_patches, utils.zoom1_labels, utils.zoom1_masks, 
        utils.zoom2_patches, utils.zoom2_labels, utils.zoom2_masks)

    # validate whether labels match
    zoom1_labels = np.load(f"{utils.zoom1_labels}/zoom1labels.npy")
    zoom2_labels = np.load(f"{utils.zoom2_labels}/zoom2labels.npy")
    print(sum(zoom1_labels == zoom2_labels) == len(zoom2_labels))