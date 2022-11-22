"""Train the model

Author: Yewen Zhou
Date: 11/4/2022
"""
import utils
import create_training_dataset
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
import model
import pickle


if __name__ == "__main__":    
    training_gen, val_gen, train_len, val_len = create_training_dataset.create_train_val_dataset()

    for patch, label in training_gen:
        print(patch[0].shape)
        print(patch[1].shape)
        print(label.shape)
        break

    mymodel = model.inceptionv3small(utils.INPUT_SIZE)
    print(mymodel.summary())
    
    mymodel.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy",
                 tfa.metrics.F1Score(num_classes=2, threshold=0.5), # https://github.com/tensorflow/addons/issues/746#issuecomment-643797601
                 keras.metrics.AUC(),
                 keras.metrics.Recall(),
                 keras.metrics.Precision(),]
    )
    
    # notes https://keras.io/api/models/model_training_apis/
    # 1. do not specify batch_size if data is generator
    # 2. need to specify steps_per_epoch
    # 3. need to specify validation_steps
    history = mymodel.fit(
        training_gen, 
        epochs=30,
        verbose=1,
        validation_data=val_gen,
        steps_per_epoch=train_len // utils.train_batch_size,
        validation_steps=val_len // utils.train_batch_size,
    )
    
    mymodel.save("./checkpoints/inceptionv3_11_22_categorical_30.h5")
    with open('./history/inceptionv3_11_22_categorical_30.pkl', 'wb') as f:
        pickle.dump(history.history, f)