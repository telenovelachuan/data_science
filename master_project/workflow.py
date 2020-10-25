import os
#os.environ["CUDA_VISIBLE_DEVICES"]="2"  # specify which GPU(s) to be used
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os.path
import math

import pandas as pd
import numpy as np
from numpy import inf
from matplotlib import pyplot as plt
import seaborn as sns
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


CTW_labelled = "~/"
def get_data(data_file):
    
    f = h5py.File(data_file, 'r')
    H_Re = f['H_Re'][:] #shape (sample size, 56, 924, 5)
    H_Im = f['H_Im'][:] #shape (sample size, 56, 924, 5)
    SNR = f['SNR'][:] #shape (sample size, 56, 5)
    Pos = f['Pos'][:] #shape(sample size, 3)
    f.close()
            
    return H_Re, H_Im, SNR, Pos        

# load data from file_1.hdf4

data_file = CTW_labelled+"file_"+str(1)+".hdf5"
H_Re, H_Im, SNR, Pos = get_data(data_file)
print("H_Re is of shape {}".format(H_Re.shape))
print("H_Im is of shape {}".format(H_Im.shape))
print("SNR is of shape {}".format(SNR.shape))
print("Pos is of shape {}".format(Pos.shape))

# Euclidean norm
H = np.sqrt(H_Re[:,:]**2 + H_Im[:,:]**2)

# Extracting phase
print("extracting phase...")
arctan = np.divide(H_Re[:,:], H_Im[:,:])
arctan[np.isnan(arctan)] = math.pi / 2
#arctan.shape

# append phase onto H
print("appending phase...")
h1 = np.empty([512, 56, 924, 10])
for d1 in range(512):
    for d2 in range(56):
        for d3 in range(924):
            h = H[d1][d2][d3]
            phase = arctan[d1][d2][d3]
            h1[d1][d2][d3] = np.vstack((h, phase)).reshape(-1)
        
#h1.shape

# append SNR onto h1
print("appending SNR...")
snr_rep = np.repeat(SNR, 2, axis=1).reshape(512, 56, -1)
h2 = np.empty([512, 56, 925, 10])
for d1 in range(512):
    for d2 in range(56):
        h = h1[d1][d2]
        snr = snr_rep[d1][d2]
        h2[d1][d2] = np.vstack((h, snr))
        
#h2.shape

# replace inifinity values
h2[h2 == inf] = 0
h2[h2 == -inf] = 0

print("Train-test split the dataset")
x_train0, x_test0, y_train0, y_test0 = train_test_split(h2, Pos, test_size=0.1, random_state=42)
#x_train0.shape, y_train0.shape

def keras_model1(opt=Adam(1e-3)):
    model = Sequential()
    model.add(Conv2D(3,(5,5), input_shape=(x_train0.shape[1:]), activation='relu'))
    model.add(Conv2D(6,(5,5), activation='relu'))
    model.add(Conv2D(8,(5,5), activation='relu'))
    model.add(Flatten())
#     model.add(Dense(8192, activation='relu'))
#     model.add(Dense(4096, activation='relu'))
#     model.add(Dense(3072, activation='relu'))
#     model.add(Dense(2048, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(3))
    model.compile(loss='mean_absolute_percentage_error', optimizer=opt) 
    return model

def keras_model3(opt=Adam(1e-3), dropout_rate=0.2):
    model = Sequential()
    model.add(Dense(960, input_shape=x_train.shape[1:], activation='relu'))
    model.add(Flatten())
    model.add(Dense(860, activation='relu'))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(560, activation='relu'))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(460, activation='relu'))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(360, activation='relu'))
    model.add(Dense(3))
    model.compile(loss='mean_squared_error', optimizer=opt)
    return model


print("training keras model3...")
model0 = keras_model3(opt=Adam(5e-3))
earlystopper = EarlyStopping(patience=50, verbose=1)
#cp = ModelCheckpoint('v1_no_pca.h5', verbose=1, save_best_only=True)
hist0 = model0.fit(x_train0, y_train0, epochs=50, validation_data=(x_test0, y_test0),
                 callbacks=[earlystopper], batch_size=4, verbose=1)
#val_loss = hist.history['val_loss']

