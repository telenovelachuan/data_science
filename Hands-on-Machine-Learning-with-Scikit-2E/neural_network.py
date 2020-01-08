import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

'''
Fashion MNIST
'''
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
print "X_train_full.shape:{}".format(X_train_full.shape)
print "X_train_full.dtype:{}".format(X_train_full.dtype)

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# model = keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=[28, 28]))
# model.add(tf.keras.layers.Dense(300, activation="relu"))
# model.add(tf.keras.layers.Dense(100, activation="relu"))
# model.add(tf.keras.layers.Dense(10, activation="softmax"))

# model = keras.models.Sequential([
#         keras.layers.Flatten(input_shape=[28, 28]),
#         keras.layers.Dense(300, activation="relu"),
#         keras.layers.Dense(100, activation="relu"),
#         keras.layers.Dense(10, activation="softmax")
# ])
# model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
# # history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))
# history = model.fit(X_train, y_train, epochs=30, validation_split=0.1)
import pandas as pd
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
plt.show()


'''
California housing price
'''
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

model = keras.models.Sequential([
        keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
        keras.layers.Dense(1)
])
model.compile(loss="mean_squared_error", optimizer="sgd")
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
mse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]  # pretend these are new instances y_pred = model.predict(X_new)


# use callback to save the best model on the validation set
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)
history = model.fit(X_train, y_train, epochs=10, callbacks=[checkpoint_cb])


# Early stopping
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), callbacks=[checkpoint_cb, early_stopping_cb])

# cutomize callback to print validation loss / training loss
class PrintValTrainRatioCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print "\nval/train: {:.2f}".format(logs["val_loss"] / logs["loss"])


# Batch Normalization after activation
model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(10, activation="softmax")
])

# Batch Normalization before activation
model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(300, kernel_initializer="he_normal", use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("elu"),
        keras.layers.Dense(100, kernel_initializer="he_normal", use_bias=False),
        keras.layers.Activation("elu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(10, activation="softmax")
])

# l1 & l2 regularization
layer = keras.layers.Dense(100, activation="elu",
                           kernel_initializer="he_normal",
                           kernel_regularizer=keras.regularizers.l2(0.01))

# Dropout
model = keras.models.Sequential([
         keras.layers.Flatten(input_shape=[28, 28]),
         keras.layers.Dropout(rate=0.2),
         keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
         keras.layers.Dropout(rate=0.2),
         keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
         keras.layers.Dropout(rate=0.2),
         keras.layers.Dense(10, activation="softmax")
])

# Monte Carlo Dropout
dropout = keras.layers.Dropout(0.5)(x, training=True)

'''
Default hyperparameters:
Kernel initializer: LeCun initialization
Activation function: SELU
Normalization: None (self-normalization)
Regularization: Early stopping
Optimizer: Nadam
Learning rate schedule: Performance scheduling
'''

