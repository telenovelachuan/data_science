import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Flatten, Dropout
from tensorflow.keras.models import Model, Sequential, model_from_json
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt
import seaborn as sns
import pickle

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

nsamples, nx, ny = x_train.shape
x_train_d2 = x_train.reshape((nsamples, nx * ny))
x_train_normed = StandardScaler().fit_transform(x_train_d2)

nsamples, nx, ny = x_test.shape
x_test_d2 = x_test.reshape((nsamples, nx * ny))
x_test_normed = StandardScaler().fit_transform(x_test_d2)

y_train_ohe = pd.get_dummies(y_train).as_matrix()
y_test_ohe = pd.get_dummies(y_test).as_matrix()

model_sequences = [
    keras.layers.Dense(units=32, activation='relu', input_shape=x_train_normed.shape[1:]),
    keras.layers.Dense(units=32, activation='relu', input_shape=x_train_normed.shape[1:]),
]

model_sequences.append(Dense(units=10, name='output', activation='softmax'))
nn_model = keras.models.Sequential(model_sequences)
print(nn_model.summary())

%%time
nn_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
history = nn_model.fit(x_train_normed, y_train_ohe, validation_data=(x_test_normed, y_test_ohe), epochs=20, batch_size=64, verbose=1)
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.show()

def save_keras_model(model, history, file_name):
    model_json = model.to_json()
    with open("./models/{}.json".format(file_name), 'w') as json_file:
        json_file.write(model_json)
    model.save_weights("./models/{}.h5".format(file_name))
    with open('./models/{}.history'.format(file_name), 'wb') as file_history:
        pickle.dump(history.history, file_history)
    print("model saved")
save_keras_model(nn_model, history, 'base_model')

model_sequences = [
    keras.layers.Dense(units=64, activation='relu', input_shape=x_train_normed.shape[1:]),
    keras.layers.Dense(units=64, activation='relu', input_shape=x_train_normed.shape[1:]),
]

model_sequences.append(Dense(units=10, name='output', activation='softmax'))
nn_model_64 = keras.models.Sequential(model_sequences)
print(nn_model_64.summary())

%%time
nn_model_64.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
history = nn_model_64.fit(x_train_normed, y_train_ohe, validation_data=(x_test_normed, y_test_ohe), epochs=20, batch_size=64, verbose=1)
save_keras_model(nn_model_64, history, 'model_64_n')
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.show()

%%time
model_sequences = [
    keras.layers.Dense(units=32, activation='relu', input_shape=x_train_normed.shape[1:]),
    keras.layers.Dense(units=32, activation='relu', input_shape=x_train_normed.shape[1:]),
]

model_sequences.append(Dense(units=10, name='output', activation='softmax'))
nn_model_lr = keras.models.Sequential(model_sequences)
sgd = SGD(lr=0.01, decay=0.000001, momentum=0.9)
nn_model_lr.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
history = nn_model_lr.fit(x_train_normed, y_train_ohe, validation_data=(x_test_normed, y_test_ohe), epochs=20, batch_size=64, verbose=1)
save_keras_model(nn_model_lr, history, 'model_lr')
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.show()

%%time
model_sequences = [
    keras.layers.Dense(units=32, activation='relu', input_shape=x_train_normed.shape[1:]),
    keras.layers.Dense(units=32, activation='relu', input_shape=x_train_normed.shape[1:]),
]

model_sequences.append(Dense(units=10, name='output', activation='softmax'))
nn_model = keras.models.Sequential(model_sequences)
nn_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
history_bs_32 = nn_model.fit(x_train_normed, y_train_ohe, validation_data=(x_test_normed, y_test_ohe), epochs=20, batch_size=32, verbose=1)
# save_keras_model(nn_model, history, 'model_64_n')
pd.DataFrame(history_bs_32.history).plot(figsize=(8, 5))
plt.grid(True)
plt.show()

%%time
model_sequences = [
    keras.layers.Dense(units=32, activation='relu', input_shape=x_train_normed.shape[1:]),
    keras.layers.Dense(units=32, activation='relu', input_shape=x_train_normed.shape[1:]),
]

model_sequences.append(Dense(units=10, name='output', activation='softmax'))
nn_model = keras.models.Sequential(model_sequences)
nn_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
history_bs_128 = nn_model.fit(x_train_normed, y_train_ohe, validation_data=(x_test_normed, y_test_ohe), epochs=20, batch_size=128, verbose=1)
# save_keras_model(nn_model, history, 'model_64_n')
pd.DataFrame(history_bs_128.history).plot(figsize=(8, 5))
plt.grid(True)
plt.show()

%%time
model_sequences = [
    keras.layers.Dense(units=32, activation='relu', input_shape=x_train_normed.shape[1:]),
    keras.layers.Dense(units=32, activation='relu', input_shape=x_train_normed.shape[1:]),
    keras.layers.Dense(units=32, activation='relu', input_shape=x_train_normed.shape[1:]),
]

model_sequences.append(Dense(units=10, name='output', activation='softmax'))
nn_model_3_layers = keras.models.Sequential(model_sequences)
nn_model_3_layers.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
history = nn_model_3_layers.fit(x_train_normed, y_train_ohe, validation_data=(x_test_normed, y_test_ohe), epochs=20, batch_size=64, verbose=1)
save_keras_model(nn_model_3_layers, history, 'model_3_layers')
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.show()

%%time
model_sequences = [
    keras.layers.Dropout(rate=0.2, input_shape=x_train_normed.shape[1:]),
    keras.layers.Dense(units=128, activation='relu', input_shape=x_train_normed.shape[1:]),
    keras.layers.Dropout(rate=0.2, input_shape=x_train_normed.shape[1:]),
    keras.layers.Dense(units=128, activation='relu', input_shape=x_train_normed.shape[1:]),
    keras.layers.Dropout(rate=0.2, input_shape=x_train_normed.shape[1:]),
]

model_sequences.append(Dense(units=10, name='output', activation='softmax'))
nn_model_dropout = keras.models.Sequential(model_sequences)
nn_model_dropout.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
history = nn_model_dropout.fit(x_train_normed, y_train_ohe, validation_data=(x_test_normed, y_test_ohe), epochs=20, batch_size=64, verbose=1)
save_keras_model(nn_model_dropout, history, 'model_128_dropout')
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.show()

%%time
act_alpha = 0.01
model_sequences = [
    keras.layers.Dropout(rate=0.03, input_shape=x_train_normed.shape[1:]),
    keras.layers.Dense(units=700,
                       activation='relu',
                       input_shape=x_train_normed.shape[1:]),
    #keras.layers.LeakyReLU(alpha=act_alpha),
    keras.layers.Dropout(rate=0.05, input_shape=x_train_normed.shape[1:]),
    keras.layers.Dense(units=700,
                       activation='relu',
                       input_shape=x_train_normed.shape[1:]),
    #keras.layers.LeakyReLU(alpha=act_alpha),
    #keras.layers.Dropout(rate=0.05, input_shape=x_train_normed.shape[1:]),
#     keras.layers.Dense(units=128,
#                        activation='relu',
#                        input_shape=x_train_normed.shape[1:]),
    #keras.layers.LeakyReLU(alpha=act_alpha),
]

model_sequences.append(Dense(units=10, name='output', activation='softmax'))
nn_model_dropout = keras.models.Sequential(model_sequences)
sgd = SGD(lr=0.01, decay=0.000001, momentum=0.9)
nn_model_dropout.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
history = nn_model_dropout.fit(x_train_normed, y_train_ohe, validation_data=(x_test_normed, y_test_ohe), epochs=20, batch_size=32, verbose=1)
save_keras_model(nn_model_dropout, history, 'model_best')
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.show()

