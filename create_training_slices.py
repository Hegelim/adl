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
from tqdm import trange
import re


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
        mask: open_slide object
        level1, level2: int, e.g. 4, 5
        center_x, center_y: coordinates of center w.r.t. level1 (not downsample 1)
        patch_size: size of the patch w.r.t. level1, int, eg 299, should be same for all levels

    Note:
        the function itself doesn't check whether the image is out of bound
        that is left for other validation functions

    Returns:
        a dictionary that has 4 k-v pairs
        slide images are 3D arrays
        mask images are 2D arrays
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
    
    level1_mask = utils.read_slide(mask, start_x_level10, start_y_level10, level1, patch_size, patch_size)[:,:,0]
    level2_mask = utils.read_slide(mask, start_x_level20, start_y_level20, level2, patch_size, patch_size)[:,:,0]
    
    res = {}
    res['level1_patch'] = level1_patch
    res['level2_patch'] = level2_patch
    res['level1_mask'] = level1_mask
    res['level2_mask'] = level2_mask
    
    return res


def not_outofbound(image) -> bool:
    """Return True if image is not out of bound
    
    Args:
        image: RGB image
    """
    # first off, sum across axis=2
    summed = np.sum(image, axis=2)
    # count how many elements sum 
    # across axis=2 == 0
    # if there's one element with RGB = [0, 0, 0]
    # that's a black pixel
    return np.sum(summed == 0) == 0


def has_enough_tissue(image, threshold=0.4) -> bool:
    """Validate the image
    No black pixels (border) and has enough tissue cells

    Args:
        image: an RGB image    
        
    Returns:
        True if image is valid
    """
    return utils.cal_tissue_perc(image) >= threshold


def find_slide_number(slide_path: str) -> int:
    m = re.search("tumor_([\d]+)", slide_path)
    if m:
        return m.group(1)
    
    
def randomly_extract_patches_to_dir(
    slide_path, mask_path, n, level1, level2, input_size, 
    zoom1_patch_normal_outdir, zoom1_patch_tumor_outdir, 
    zoom1_mask_normal_outdir, zoom1_mask_tumor_outdir,
    zoom2_patch_normal_outdir, zoom2_patch_tumor_outdir,
    zoom2_mask_normal_outdir, zoom2_mask_tumor_outdir) -> None:
    """Extract patches from im to dir
    1. Assume we will extract patches from the entire image at 2 zoom levels
    2. focus on level1
    3. randomly select indices that belong to normal/tumor
    
    Args:
        n: number of samples per each category

    """
    def extract_save(indices, sample_size, is_tumor: bool):
        counter = 1
        for i in trange(len(indices)):
            if counter > sample_size:
                return
            
            center_x, center_y = indices[i]
            patch_mask = extract_patch_from_center(slide, mask, center_x, center_y, level1, level2, input_size)
            patch_level1, patch_level2 = patch_mask['level1_patch'], patch_mask['level2_patch']
            mask_level1, mask_level2 = patch_mask['level1_mask'], patch_mask['level2_mask']
            
            if (has_enough_tissue(patch_level1) and has_enough_tissue(patch_level2) and 
                not_outofbound(patch_level1) and not_outofbound(patch_level2)):
                
                out_images = [patch_level1, mask_level1, patch_level2, mask_level2]
                if is_tumor:
                    for path, image in list(zip(tumor_out_paths, out_images)):
                        if image.ndim == 3:
                            cv2.imwrite(f"{path}/{slide_number}_tumor_{counter}.png", image)
                        # if it's mask
                        elif image.ndim == 2:
                            np.save(f"{path}/{slide_number}_tumor_{counter}.npy", image)
                else:
                    for path, image in list(zip(normal_out_paths, out_images)):
                        if image.ndim == 3:
                            cv2.imwrite(f"{path}/{slide_number}_normal_{counter}.png", image)
                        elif image.ndim == 2:
                            np.save(f"{path}/{slide_number}_normal_{counter}.npy", image)
                counter += 1
        return
        
    # level1 - zoomed-in version
    # level2 - zoomed-out version
    assert level1 < level2
    
    slide = open_slide(slide_path)
    mask = open_slide(mask_path)
    slide_number = find_slide_number(slide_path)
    # a 2D array with 0s and 1s
    mask_level1_im = utils.read_entire_slide(mask, level1)[:,:,0]
    # indices of tumor (those indices are (center_x, center_y))
    tumor_indices = np.transpose(np.where(mask_level1_im == 1))
    normal_indices = np.transpose(np.where(mask_level1_im == 0))
    np.random.shuffle(tumor_indices)
    np.random.shuffle(normal_indices)
    assert n < len(tumor_indices)
    
    tumor_out_paths = [zoom1_patch_tumor_outdir, zoom1_mask_tumor_outdir, zoom2_patch_tumor_outdir, zoom2_mask_tumor_outdir]
    normal_out_paths = [zoom1_patch_normal_outdir, zoom1_mask_normal_outdir, zoom2_patch_normal_outdir, zoom2_mask_normal_outdir]
    
    # since we already know whether the center is a 
    # tumor or not, we don't need to extract the labels again
    extract_save(tumor_indices, n, True)
    extract_save(normal_indices, n, False)
        
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
        utils.zoom1_tumor_patches,
        utils.zoom1_normal_patches,
        utils.zoom1_tumor_masks,
        utils.zoom1_normal_masks,
        utils.zoom2_tumor_patches,
        utils.zoom2_normal_patches,
        utils.zoom2_tumor_masks,
        utils.zoom2_normal_masks,
    ]
    
    # set rerun=True to reset directories
    make_dirs(dirs, rerun=True)

    randomly_extract_patches_to_dir(
        utils.train_input_im, utils.train_input_mask, utils.TRAIN_SAMPLE_SIZE,
        utils.level1, utils.level2, utils.INPUT_SIZE,
        zoom1_patch_normal_outdir=utils.zoom1_normal_patches,
        zoom1_patch_tumor_outdir=utils.zoom1_tumor_patches,
        zoom1_mask_normal_outdir=utils.zoom1_normal_masks,
        zoom1_mask_tumor_outdir=utils.zoom1_tumor_masks,
        zoom2_patch_normal_outdir=utils.zoom2_normal_patches,
        zoom2_patch_tumor_outdir=utils.zoom2_tumor_patches,
        zoom2_mask_normal_outdir=utils.zoom2_normal_masks,
        zoom2_mask_tumor_outdir=utils.zoom2_tumor_masks)
    