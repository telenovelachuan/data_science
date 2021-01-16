import os
# os.environ["CUDA_VISIBLE_DEVICES"]="2"  # specify which GPU(s) to be used
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os.path
import math

import pandas as pd
from numpy import inf
from matplotlib import pyplot as plt
import seaborn as sns
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Activation, AveragePooling2D, Conv2DTranspose, ZeroPadding2D
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, Conv1D, BatchNormalization
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from modAL.models import ActiveLearner
from modAL.uncertainty import entropy_sampling, margin_sampling, uncertainty_sampling
import copy

CTW_labelled = "/home/c693s270/"

def get_data(data_file):
    f = h5py.File(data_file, 'r')
    H_Re = f['H_Re'][:]  # shape (sample size, 56, 924, 5)
    H_Im = f['H_Im'][:]  # shape (sample size, 56, 924, 5)
    SNR = f['SNR'][:]  # shape (sample size, 56, 5)
    Pos = f['Pos'][:]  # shape(sample size, 3)
    f.close()
    return H_Re, H_Im, SNR, Pos

# load data from pre-saved file
print("Loading all training data...")
with open('all_ae.npy', 'rb') as f:
    # 4979 * 56 * 925 * 10
    h2_ae = np.load(f)
    Pos = np.load(f)

def cnn_model1(input_shape, opt=Adam(1e-3), dropout_rate=0.2):
    model = Sequential()
    model.add(Conv1D(16, 16, input_shape=input_shape, activation='relu'))
    model.add(Conv1D(32, 16, activation='relu'))
    model.add(Conv1D(32, 16, activation='relu'))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(3))
    model.compile(loss='mean_squared_error', optimizer=opt)
    return model

h2_ae_shape = h2_ae.shape
h2_ae_1dcnn = h2_ae.reshape(h2_ae_shape[0], h2_ae_shape[1], -1)

print("train-test splitting ae data...")
x_train_ae_cnn, x_test_ae_cnn, y_train_ae_cnn, y_test_ae_cnn = train_test_split(h2_ae_1dcnn, Pos, test_size=0.1, random_state=42)

# shuffle the original training data
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
x_train_ae_cnn, y_train_ae_cnn = unison_shuffled_copies(x_train_ae_cnn, y_train_ae_cnn)

print("constructing 1D-CNN model...")
lr = 5e-3
epochs = 50
decay_rate = lr / 50
model_ae = cnn_model1(x_train_ae_cnn.shape[1:], opt=Adam(5e-3, decay=decay_rate))

def get_prediction_precision(regressor, x_test, y_test):
    y_pred = regressor.predict(x_test)
    mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    #print(f"current MSE: {mse}")
    return mse

# Full fit on original training data
print(f"x_train_ae_cnn:{x_train_ae_cnn.shape}, y_train_ae_cnn:{y_train_ae_cnn.shape}")
model_ae.fit(x_train_ae_cnn, y_train_ae_cnn, epochs=34, validation_data=(x_test_ae_cnn, y_test_ae_cnn), verbose=1)
print("Finished full training, evaluating...")
mse = get_prediction_precision(model_ae, x_test_ae_cnn, y_test_ae_cnn)
print(f"final MSE: {mse}")
print("saving predictions...")
preds = model_ae.predict(x_test_ae_cnn)
result = pd.DataFrame(preds, columns=["x", "y", "z"])
result.to_csv(f"preds/cnn_ae_full_all_preds.csv")



