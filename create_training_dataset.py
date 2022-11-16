"""Create Training Dataset from slices

Author: Yewen Zhou
Date: 11/3/2022
"""
import utils
from keras.preprocessing.image import ImageDataGenerator


def get_generator(train_zoom1, train_zoom2):
    """_summary_

    Args:
        train_zoom1 (ImageDataGenerator): on level1
        train_zoom2 (ImageDataGenerator): on level2
    """
    while True:
        zoom1_patch, zoom1_label = next(train_zoom1)
        zoom2_patch, zoom2_label = next(train_zoom2)
        yield (zoom1_patch, zoom2_patch), zoom1_label


def create_train_val_dataset():
    image_generator = ImageDataGenerator(
        rescale=1/255,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2,
    )
    
    train_zoom1 = image_generator.flow_from_directory(
        batch_size=utils.train_batch_size,
        directory=utils.zoom1_patches,
        target_size=(utils.INPUT_SIZE, utils.INPUT_SIZE),
        color_mode="rgb",
        class_mode="binary",
        shuffle=True,
        seed=utils.seed,
        subset="training",
    )
    
    val_zoom1 = image_generator.flow_from_directory(
        batch_size=utils.train_batch_size,
        directory=utils.zoom1_patches,
        target_size=(utils.INPUT_SIZE, utils.INPUT_SIZE),
        color_mode="rgb",
        class_mode="binary",
        shuffle=True,
        seed=utils.seed,
        subset="validation",
    )
    
    train_zoom2 = image_generator.flow_from_directory(
        batch_size=utils.train_batch_size,
        directory=utils.zoom2_patches,
        target_size=(utils.INPUT_SIZE, utils.INPUT_SIZE),
        color_mode="rgb",
        class_mode="binary",
        shuffle=True,
        seed=utils.seed,
        subset="training",
    )
    
    val_zoom2 = image_generator.flow_from_directory(
        batch_size=utils.train_batch_size,
        directory=utils.zoom2_patches,
        target_size=(utils.INPUT_SIZE, utils.INPUT_SIZE),
        color_mode="rgb",
        class_mode="binary",
        shuffle=True,
        seed=utils.seed,
        subset="validation",
    )
    
    training_generator = get_generator(train_zoom1, train_zoom2)
    val_generator = get_generator(val_zoom1, val_zoom2)
    
    return training_generator, val_generator, len(train_zoom1), len(val_zoom1)