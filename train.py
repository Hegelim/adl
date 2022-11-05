"""Train the model

Author: Yewen Zhou
Date: 11/4/2022
"""
import utils
import create_training_dataset
import tensorflow as tf
from tensorflow import keras
import model


if __name__ == "__main__":    
    training_gen, val_gen, train_len, val_len = create_training_dataset.create_train_val_dataset()

    mymodel = model.customized_model(utils.INPUT_SIZE)
    print(mymodel.summary())
    
    mymodel.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.Accuracy(),
                 keras.metrics.AUC(),
                 keras.metrics.Recall()]
    )
    
    # notes https://keras.io/api/models/model_training_apis/
    # 1. do not specify batch_size if data is generator
    # 2. need to specify steps_per_epoch
    # 3. need to specify validation_steps
    history = mymodel.fit(
        training_gen, 
        epochs=20,
        verbose=1,
        validation_data=val_gen,
        steps_per_epoch=train_len // utils.train_batch_size,
        validation_steps=val_len // utils.train_batch_size,
    )