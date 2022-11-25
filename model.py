"""Create models"""
import utils
import tensorflow as tf
from tensorflow import keras


def inceptionv3small(input_size=utils.INPUT_SIZE):
    zoom1_model = tf.keras.applications.InceptionV3(input_shape=(input_size, input_size, 3), 
                                                    include_top=False, 
                                                    weights='imagenet')
    zoom1_model.trainable = False
    
    zoom2_model = tf.keras.applications.InceptionV3(input_shape=(input_size, input_size, 3), 
                                                    include_top=False, 
                                                    weights='imagenet')
    zoom2_model.trainable = False
    
    zoom1_input = keras.layers.Input(shape=(input_size, input_size, 3))
    zoom2_input = keras.layers.Input(shape=(input_size, input_size, 3))
    
    zoom1_model = keras.Sequential([
        zoom1_model,
        keras.layers.GlobalAveragePooling2D(),
    ])
    
    zoom2_model = keras.Sequential([
        zoom2_model,
        keras.layers.GlobalAveragePooling2D()
    ])
    
    res_zoom1 = zoom1_model(zoom1_input)
    res_zoom2 = zoom2_model(zoom2_input)
    merged = keras.layers.concatenate([res_zoom1, res_zoom2])
    
    dense1 = keras.layers.Dense(512, activation="relu")(merged)
    dropout1 = keras.layers.Dropout(0.5)(dense1)
    dense2 = keras.layers.Dense(256, activation="relu")(dropout1)
    dropout2 = keras.layers.Dropout(0.5)(dense2)
    dense3 = keras.layers.Dense(128, activation="relu")(dropout2)
    output = keras.layers.Dense(2, activation="softmax")(dense3)
    model = keras.Model(inputs=[zoom1_input, zoom2_input], outputs=output, name="myModel")
    return model


def customized_model(input_size=utils.INPUT_SIZE):
    zoom1_input = keras.layers.Input(shape=(input_size, input_size, 3))
    zoom2_input = keras.layers.Input(shape=(input_size, input_size, 3))
    
    zoom1_model = keras.Sequential([
        keras.layers.Conv2D(16, kernel_size=3, strides=(1, 1), activation="relu"),
        keras.layers.Conv2D(16, kernel_size=3, strides=(1, 1), activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, kernel_size=3, strides=(1, 1), activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, kernel_size=3, strides=(1, 1), activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(128, kernel_size=3, strides=(1, 1), activation="relu"),
        keras.layers.MaxPooling2D(),
    ])
    
    zoom2_model = keras.Sequential([
        keras.layers.Conv2D(16, kernel_size=3, activation="relu"),
        keras.layers.Conv2D(16, kernel_size=3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, kernel_size=3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, kernel_size=3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(128, kernel_size=3, activation="relu"),
        keras.layers.MaxPooling2D(),
    ])
    
    res_zoom1 = zoom1_model(zoom1_input)
    res_zoom2 = zoom2_model(zoom2_input)
    merged = keras.layers.concatenate([res_zoom1, res_zoom2])
    
    dense1 = keras.layers.Dense(256, activation="relu")(merged)
    dropout1 = keras.layers.Dropout(0.5)(dense1)
    dense2 = keras.layers.Dense(128, activation="relu")(dropout1)
    dropout2 = keras.layers.Dropout(0.5)(dense2)
    dense3 = keras.layers.Dense(32, activation="relu")(dropout2)
    flatten = keras.layers.Flatten()(dense3)
    output = keras.layers.Dense(2, activation="softmax")(flatten)
    model = keras.Model(inputs=[zoom1_input, zoom2_input], outputs=output, name="myModel")
    return model