"""Create models"""
import utils
import tensorflow as tf
from tensorflow import keras


def customized_model(input_size=utils.INPUT_SIZE):
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
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=32, kernel_size=3, strides=3, padding="same"),
        keras.layers.MaxPooling2D(padding="same"),
        keras.layers.Conv2D(filters=64, kernel_size=3, strides=3, padding="same"),
        keras.layers.MaxPooling2D(padding="same"),
        keras.layers.Conv2D(filters=128, kernel_size=3, strides=3, padding="same"),
        keras.layers.MaxPooling2D(padding="same"),
    ])
    
    zoom2_model = keras.Sequential([
        zoom2_model,
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=32, kernel_size=3, strides=3, padding="same"),
        keras.layers.MaxPooling2D(padding="same"),
        keras.layers.Conv2D(filters=64, kernel_size=3, strides=3, padding="same"),
        keras.layers.MaxPooling2D(padding="same"),
        keras.layers.Conv2D(filters=128, kernel_size=3, strides=3, padding="same"),
        keras.layers.MaxPooling2D(padding="same"),
    ])
    
    res_zoom1 = zoom1_model(zoom1_input)
    res_zoom2 = zoom2_model(zoom2_input)
    merged = keras.layers.concatenate([res_zoom1, res_zoom2])
    
    dense2 = keras.layers.Dense(16, activation="relu")(merged)
    dropout1 = keras.layers.Dropout(0.5)(dense2)
    flatten = keras.layers.Flatten()(dropout1)
    output = keras.layers.Dense(1, activation="sigmoid")(flatten)
    model = keras.Model(inputs=[zoom1_input, zoom2_input], outputs=output, name="myModel")
    return model


def vggsmall(input_size=utils.INPUT_SIZE):
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
    
    dense1 = keras.layers.Dense(256, activation="relu")(merged)
    dropout1 = keras.layers.Dropout(0.5)(dense1)
    dense2 = keras.layers.Dense(128, activation="relu")(dropout1)
    output = keras.layers.Dense(2, activation="softmax")(dense2)
    model = keras.Model(inputs=[zoom1_input, zoom2_input], outputs=output, name="myModel")
    return model


def customized_model_3(input_size=utils.INPUT_SIZE):
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
    dense2 = keras.layers.Dense(32, activation="relu")(dropout1)
    flatten = keras.layers.Flatten()(dense2)
    output = keras.layers.Dense(2, activation="softmax")(flatten)
    model = keras.Model(inputs=[zoom1_input, zoom2_input], outputs=output, name="myModel")
    return model