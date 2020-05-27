import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
import json
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:12345", "localhost:23456"]
    },
    'task': {'type': 'worker', 'index': 1}
})
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
checkpoint_dir = "./ckpt"
TensorBoard_dir = './tensorboard'


def mnist_dataset(batch_size):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    # The `x` arrays are in uint8 and have values in the range [0, 255].
    # We need to convert them to float32 with values in the range [0, 1]
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
    return train_dataset


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


per_worker_batch_size = 64
num_workers = 4

# Here the batch size scales up by number of workers since
# `tf.data.Dataset.batch` expects the global batch size. Previously we used 64,
# and now this becomes 128.
global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = mnist_dataset(global_batch_size)

with strategy.scope():
    # Model building/compiling need to be within `strategy.scope()`.
    multi_worker_model = make_or_restore_model()

callbacks = [
    # This callback saves a SavedModel every 100 batches
    keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,
                                    save_freq=30),
    keras.callbacks.TensorBoard(TensorBoard_dir)
]
# Keras' `model.fit()` trains the model with specified number of epochs and
# number of steps per epoch. Note that the numbers here are for demonstration
# purposes only and may not sufficiently produce a model with good quality.
multi_worker_model.fit(multi_worker_dataset, callbacks=callbacks, epochs=9, steps_per_epoch=70)
