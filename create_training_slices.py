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
import glob
from skimage.color import rgb2gray
import gc

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
        or an empty dictionary
    """
    assert level1 < level2 
    level1_downsample_fac = int(slide.level_downsamples[level1])
    level2_downsample_fac = int(slide.level_downsamples[level2])
    
    start_x_level1 = center_x - patch_size // 2
    start_y_level1 = center_y - patch_size // 2

    start_x_level10 = start_x_level1 * level1_downsample_fac
    start_y_level10 = start_y_level1 * level1_downsample_fac

    center_x_level2 = center_x * level1_downsample_fac // level2_downsample_fac # /2
    center_y_level2 = center_y * level1_downsample_fac // level2_downsample_fac

    start_x_level2 = center_x_level2 - patch_size // 2
    start_y_level2 = center_y_level2 - patch_size // 2

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


def has_enough_tissue(image, threshold=0.2) -> bool:
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
    
    
def extract_patches_to_dir(
    training_path, level1, level2, input_size, sample_size,
    zoom1_patch_normal_outdir, zoom1_patch_tumor_outdir, 
    zoom1_mask_normal_outdir, zoom1_mask_tumor_outdir,
    zoom2_patch_normal_outdir, zoom2_patch_tumor_outdir,
    zoom2_mask_normal_outdir, zoom2_mask_tumor_outdir) -> None:
    """Extract patches from im to dir
    1. Assume we will extract patches from the entire image at 2 zoom levels
    2. focus on level1
    3. randomly select indices that belong to normal/tumor
    
    Failed Implementation:
    1. slide through the image: too slow
    2. not specifying sample size: run through entire image, slow
    
    Args:
        n: number of samples per each category

    """
    # level1 - zoomed-in version
    # level2 - zoomed-out version
    def save_patch(indices, tumor_count, normal_count, total_tumor_count, total_normal_count):
        center_y, center_x = indices[i] # x, y based on computer vision
        
        patch_mask = extract_patch_from_center(slide, mask, center_x, center_y, level1, level2, input_size)
        patch_level1, patch_level2 = patch_mask['level1_patch'], patch_mask['level2_patch']
        mask_level1, mask_level2 = patch_mask['level1_mask'], patch_mask['level2_mask']
        
        have_enough_tissue = np.count_nonzero(rgb2gray(patch_level1) <= 0.8) >= (0.3 * input_size ** 2)
        
        if (have_enough_tissue and not_outofbound(patch_level1) and not_outofbound(patch_level2)):
            
            out_images = [patch_level1, mask_level1, patch_level2, mask_level2]
            if utils.has_tumor_at_center(mask_level1):
                for path, image in list(zip(tumor_out_paths, out_images)):
                    if image.ndim == 3:
                        cv2.imwrite(f"{path}/{slide_number}_tumor_{tumor_count}.png", image)
                    # if it's mask
                    elif image.ndim == 2:
                        np.save(f"{path}/{slide_number}_tumor_{tumor_count}.npy", image)
                tumor_count += 1
                total_tumor_count += 1
            else:
                for path, image in list(zip(normal_out_paths, out_images)):
                    if image.ndim == 3:
                        cv2.imwrite(f"{path}/{slide_number}_normal_{normal_count}.png", image)
                    elif image.ndim == 2:
                        np.save(f"{path}/{slide_number}_normal_{normal_count}.npy", image)
                normal_count += 1
                total_normal_count += 1
        
        res = {}
        res["tumor_count"] = tumor_count
        res["normal_count"] = normal_count
        res["total_tumor_count"] = total_tumor_count
        res["total_normal_count"] = total_normal_count
        return res

    assert level1 < level2
    
    tumor_out_paths = [zoom1_patch_tumor_outdir, zoom1_mask_tumor_outdir, zoom2_patch_tumor_outdir, zoom2_mask_tumor_outdir]
    normal_out_paths = [zoom1_patch_normal_outdir, zoom1_mask_normal_outdir, zoom2_patch_normal_outdir, zoom2_mask_normal_outdir]
    
    print(f"Scanning {training_path}...")
    slide_files, mask_files = read_slide_tumor_tifs(training_path)
    print(f"Find {len(slide_files)} slides...")
    
    total_tumor_count, total_normal_count = 0, 0
    for s, m in zip(slide_files, mask_files):
        slide = open_slide(s)
        mask = open_slide(m)
        slide_number = find_slide_number(s)
        if slide_number == "064":
            print(f"Processing slide {slide_number}...")

            mask_level1_im = utils.read_entire_slide(mask, level1)[:,:,0]
            slide_gray = rgb2gray(utils.read_entire_slide(slide, level1))
            
            tumor_indices = np.transpose(np.where((mask_level1_im == 1)))[::20]
            normal_indices = np.transpose(np.where((mask_level1_im == 0) & (slide_gray <= 0.8)))
            after_start_rows = np.where((normal_indices[:,0] > input_size//2) & (normal_indices[:,1] > input_size//2))
            normal_indices = normal_indices[after_start_rows]
            normal_indices = normal_indices[len(normal_indices)//4:len(normal_indices)*3//4:1000]
            
            print("Finished generating indices! Now begin cutting slices...")

            tumor_count, normal_count = 0, 0
            for i in trange(len(tumor_indices)):
                if tumor_count < sample_size:
                    res = save_patch(tumor_indices, tumor_count, normal_count, total_tumor_count, total_normal_count)
                    tumor_count = res["tumor_count"]
                    normal_count = res["normal_count"]
                    total_tumor_count = res["total_tumor_count"]
                    total_normal_count = res["total_normal_count"]
                else:
                    break
                
            for i in trange(len(normal_indices)):
                if normal_count < sample_size:
                    res = save_patch(normal_indices, tumor_count, normal_count, total_tumor_count, total_normal_count)
                    tumor_count = res["tumor_count"]
                    normal_count = res["normal_count"]
                    total_tumor_count = res["total_tumor_count"]
                    total_normal_count = res["total_normal_count"]
                else:
                    break
        
            gc.collect()
            print(f"Finished slide {slide_number}!")
            print(f"Processed {tumor_count} tumor slides and {normal_count} normal slides")

            break
        
    print("All finished!")
    print(f"Finished {total_tumor_count} patches")
    print(f"Finished {total_normal_count} patches")
    
    
def read_slide_tumor_tifs(input_dir):
    files = glob.glob(f"{input_dir}/*.tif")
    slides, masks = [], []
    for f in files:
        if os.path.splitext(f)[0].endswith("mask"):
            masks.append(f)
        else:
            slides.append(f)
    slides.sort()
    masks.sort()
    return slides, masks
    

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
    # make_dirs(dirs, rerun=True)
    
    args = {
        "training_path": utils.train_input,
        "level1": utils.level1,
        "level2": utils.level2,
        "input_size": utils.INPUT_SIZE,
        "sample_size": utils.TRAIN_SAMPLE_SIZE,
        "zoom1_patch_normal_outdir": utils.zoom1_normal_patches,
        "zoom1_patch_tumor_outdir": utils.zoom1_tumor_patches,
        "zoom1_mask_normal_outdir": utils.zoom1_normal_masks, 
        "zoom1_mask_tumor_outdir": utils.zoom1_tumor_masks,
        "zoom2_patch_normal_outdir": utils.zoom2_normal_patches, 
        "zoom2_patch_tumor_outdir": utils.zoom2_tumor_patches,
        "zoom2_mask_normal_outdir": utils.zoom2_normal_masks, 
        "zoom2_mask_tumor_outdir": utils.zoom2_tumor_masks,
    }

    extract_patches_to_dir(**args)
    