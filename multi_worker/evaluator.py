import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import json

strategy = tf.distribute.MirroredStrategy()
checkpoint_dir = "./ckpt"
TensorBoard_dir = './tensorboard'


def build_and_compile_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28)),
        tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=['accuracy'])
    return model


def make_or_restore_model():
    # Either restore the latest model, or create a fresh one if there is no checkpoint available.
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    return build_and_compile_cnn_model()


with strategy.scope():
  model = make_or_restore_model()  # Restore from the checkpoint saved by the chief.

results = model.evaluate(val_dataset)