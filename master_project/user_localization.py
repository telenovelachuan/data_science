#!/usr/bin/env python
# coding: utf-8

# Import libraries

# In[19]:


import os
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
from tensorflow.keras.layers import Input, Activation, AveragePooling2D, Conv2DTranspose, ZeroPadding2D
from tensorflow.keras.layers import Dense , Dropout, Conv2D ,Flatten, Conv1D, BatchNormalization
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# Read input file

# In[5]:


CTW_labelled = "/Users/chuansun/Downloads/CTW2020_labelled_data/"


# In[6]:


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


# Extract amplitude and phase from the real & imaginary parts

# In[7]:


# Euclidean norm
H = np.sqrt(H_Re[:,:]**2 + H_Im[:,:]**2)

# Extracting phase
arctan = np.divide(H_Re[:,:], H_Im[:,:])
arctan[np.isnan(arctan)] = math.pi / 2
arctan.shape


# Append phase onto H

# In[9]:


h1 = np.empty([512, 56, 924, 10])
for d1 in range(512):
    for d2 in range(56):
        for d3 in range(924):
            h = H[d1][d2][d3]
            phase = arctan[d1][d2][d3]
            h1[d1][d2][d3] = np.vstack((h, phase)).reshape(-1)
        
h1.shape


# Append SNR onto h1

# In[10]:


# append SNR onto h1
snr_rep = np.repeat(SNR, 2, axis=1).reshape(512, 56, -1)
h2 = np.empty([512, 56, 925, 10])
for d1 in range(512):
    for d2 in range(56):
        h = h1[d1][d2]
        snr = snr_rep[d1][d2]
        h2[d1][d2] = np.vstack((h, snr))
        
h2.shape


# Replace inifinity values

# In[11]:


# replace inifinity values
h2[h2 == inf] = 0
h2[h2 == -inf] = 0


# In[20]:


# Take a look at value distribution
flatten = h2.reshape(-1)
plt.boxplot(flatten, vert=False, showfliers=False)


# In[12]:


# Train test split raw data
x_train0, x_test0, y_train0, y_test0 = train_test_split(h2, Pos, test_size=0.1, random_state=42)
x_train0.shape, y_train0.shape


# Test NN models

# In[7]:


# 2-D conv, for benchmarking only
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

def keras_model2(opt=Adam(1e-3)):
    model = Sequential()
    model.add(Conv2D(32,(3,3),activation='linear', padding='same',input_shape=x_train.shape[1:]))
    model.add(Conv2D(16,(3,3),activation='linear', padding='same'))
    model.add(Flatten())
    model.add(Dense(512 ,activation = 'relu'))
    model.add(Dense(256 ,activation = 'relu' ))
    model.add(Dense(128 ,activation = 'relu' ))
    model.add(Dense(3))
    model.compile(loss="mean_absolute_percentage_error", optimizer=opt) 
    return model


# In[161]:


model0 = keras_model1(opt=Adam(1e-3))
earlystopper = EarlyStopping(patience=50, verbose=1)
#cp = ModelCheckpoint('v1_no_pca.h5', verbose=1, save_best_only=True)
hist0 = model0.fit(x_train0, y_train0, epochs=1, validation_data=(x_test0, y_test0),
                 callbacks=[earlystopper], batch_size=4, verbose=1)
#val_loss = hist.history['val_loss']


# Modeling after PCA

# Try to explore the optimal number for PCA/other DR techniques as well

# In[133]:


# if reduce dimension to 100
condense_to = 100
pca = PCA(n_components=condense_to)
pca.fit_transform(h2.reshape(512, -1))
plt.figure(figsize=(20, 10))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Explained variance ratio")
plt.title("Explained Variance")
plt.xticks(np.arange(0, condense_to, 1))


# In[140]:


cumsum = np.cumsum(pca.explained_variance_ratio_)
cumsum[55]


# In[22]:


# reduce to 55 dimensions for every antenna array
condense_to = 55
h2_pca = np.empty((h2.shape[0], h2.shape[1], condense_to, h2.shape[3]))


evrs = []
for i in range(h2.shape[1]):
    for j in range(h2.shape[3]):
        pca = PCA(n_components=condense_to)
        h2_pca[:, i, :, j] = pca.fit_transform(h2[:, i, :, j])
        evr = np.cumsum(pca.explained_variance_ratio_)
        evrs.append(evr)
        

df_evrs = pd.DataFrame(evrs, columns=[f"p{x}" for x in range(len(evrs[0]))])
avg_evrs = []
for col in df_evrs.columns:
    avg_evrs.append(df_evrs[col].mean())

plt.figure(figsize=(20, 10))
plt.plot(avg_evrs)
plt.xlabel("Number of Components")
plt.ylabel("Explained variance ratio")
plt.title("Explained Variance")
plt.xticks(np.arange(0, len(avg_evrs), 1))


# In[28]:


# Train test split after PCA
x_train, x_test , y_train , y_test = train_test_split(h2_pca, Pos, test_size=0.1, random_state=42)
x_train.shape, y_train.shape


# In[59]:


def keras_model2(opt=Adam(1e-3)):
    model = Sequential()
    model.add(Conv2D(32,(3,3),activation='elu', padding='same', input_shape=x_train.shape[1:]))
    model.add(Conv2D(16,(3,3),activation='elu', padding='same'))
    model.add(Conv2D(8,(3,3),activation='elu', padding='same'))
    model.add(Flatten())
    model.add(Dense(512 ,activation='relu'))
    model.add(Dense(256 ,activation='relu'))
    model.add(Dense(128 ,activation='relu'))
    model.add(Dense(64 ,activation='relu'))
    model.add(Dense(3))
    model.compile(loss="mean_absolute_percentage_error", optimizer=opt) 
    return model

model = keras_model2(opt=Adam(5e-3))
earlystopper = EarlyStopping(patience=50, verbose=1)
#cp = ModelCheckpoint('v1_no_pca.h5', verbose=1, save_best_only=True)
hist2 = model.fit(x_train, y_train , epochs=50, validation_data=(x_test, y_test),
                 callbacks=[earlystopper], batch_size=4, verbose=1)
pd.DataFrame(hist2.history).plot(figsize=(8, 5))
plt.grid(True)
plt.show()


# The model below applied some adjustments mentioned during meeting, including:
# - dropouts & BN, dense layers only(without PCA)
# - normalization
# - MSE error
# - learning rate schedule
# - try batch sizes

# In[ ]:


def keras_model3(opt=Adam(1e-3), dropout_rate=0.2):
    model = Sequential()
    model.add(Dense(512, input_shape=x_train0.shape[1:], activation='relu'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=dropout_rate))
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(rate=dropout_rate))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dropout(rate=dropout_rate))
#     model.add(Dense(16, activation='relu'))
    model.add(Dense(3))
    model.compile(loss='mean_squared_error', optimizer=opt)
    return model

def dnn_model(opt=Adam(1e-3), dropout_rate=0.2):
    model = Sequential()
    model.add(Dense(512, input_shape=x_train.shape[1:], activation='relu'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(3))
    model.compile(loss='mean_squared_error', optimizer=opt)
    return model
    
model = keras_model3(opt=Adam(5e-3), dropout_rate=0.2)
earlystopper = EarlyStopping(patience=50, verbose=1)
#cp = ModelCheckpoint('v1_no_pca.h5', verbose=1, save_best_only=True)
hist1 = model.fit(x_train, y_train , epochs=10, validation_data=(x_test, y_test),
                  callbacks=[earlystopper], batch_size=4, verbose=1)
pd.DataFrame(hist1.history).plot(figsize=(8, 5))
plt.grid(True)
plt.show()


# Train the model on ITTC lab machine, and copy predicted file back here for analyzing

# In[25]:


# analyze predictions
def analyze_predictions(file, y_test):

    preds = pd.read_csv(f"preds/{file}")
    preds = preds[["x", "y", "z"]]
    preds["type"] = "pred"
    truth = pd.DataFrame(y_test, columns=["x", "y", "z"])
    truth = truth[["x", "y", "z"]]
    truth["type"] = "truth"
    concat = pd.concat([preds, truth], ignore_index=True)
    sns.set()
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x="x", y="y", data=concat, hue="type")

#analyze_predictions("model3_preds.csv")


# In[29]:


analyze_predictions("dnn_bs32_preds.csv", y_test)


# In[46]:


analyze_predictions("cnn1_preds.csv")


# In[51]:


# 1D-CNN with batch normalization
analyze_predictions("cnn1_bn_preds.csv")


# Try AutoEncoder

# In[18]:


def data_gen(data):
    for i in range(len(data)):
        yield (data[i: i + 1], data[i: i + 1])


# In[36]:


def autoencoder():
    model = Sequential()
    model.add(Conv2D(128, (5, 5),input_shape=h2.shape[1:], activation='relu', padding='same'))
    model.add(AveragePooling2D((1, 5)))
    model.add(Conv2D(32, (3, 3),activation='relu', padding='same'))
    model.add(AveragePooling2D((1, 5)))
    model.add(Conv2D(32,(3,3),activation='linear', padding='same'))
    model.add(Conv2DTranspose(32, (3,3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2DTranspose(128,(3,3), strides=(1, 5), padding='same', activation='relu'))
    #model.add(ZeroPadding2D(((0, 0), (2, 1))))
    model.add(Conv2DTranspose(10, (3, 3) , strides=(1, 5), padding='same', activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=Adam(1e-3)) 
    return model

ae = autoencoder()
# ae.summary()


# In[37]:


ae.summary()


# In[25]:


# Make training data for autoencoder. Labels are training set itself
data, data_v  = train_test_split(h2, test_size=0.5, random_state=54) 
data.shape


# In[39]:


# Train autoencoder
for i in range(5):
    ae.fit_generator(data_gen(data),validation_data=data_gen(data_v), epochs=1, steps_per_epoch=len(data),
                     validation_steps=len(data_v))


# In[43]:


# Extract the first half of the model as encoder
encoder = Model(ae.input, ae.layers[-5].output)
encoder.summary()


# In[44]:


# Get the dimension-reduced dataset using the encoder
h2_ae = encoder.predict(h2)
h2_ae.shape


# In[45]:


# Train test split the autoencoded dataset
x_train_ae, x_test_ae, y_train_ae, y_test_ae = train_test_split(h2_ae, Pos, test_size=0.1, random_state=42)


# In[48]:


# DNN & 1-D CNN models for autoencoded training dataset
def dnn_ae_model(opt=Adam(1e-3), dropout_rate=0.2):
    model = Sequential()
    model.add(Dense(512, input_shape=x_train_ae.shape[1:], activation='relu'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(3))
    model.compile(loss='mean_squared_error', optimizer=opt)
    return model

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

print("training dnn model on ae...")
lr = 5e-3
epochs = 50
decay_rate = lr / 50
model_ae = cnn_model1(x_train_ae_cnn.shape[1:], opt=Adam(5e-3, decay=decay_rate))
earlystopper = EarlyStopping(patience=50, verbose=1)
#cp = ModelCheckpoint('v1_no_pca.h5', verbose=1, save_best_only=True)
hist_ae = model_ae.fit(x_train_ae_cnn, y_train_ae_cnn, epochs=50, validation_data=(x_test_ae_cnn, y_test_ae_cnn),
                 callbacks=[earlystopper], batch_size=4, verbose=1)


# In[57]:


# DNN with autoencoder
analyze_predictions("dnn_bs32_ae_preds.csv", y_test_ae)


# In[59]:


# 1-D CNN with autoencoder
analyze_predictions("cnn_ae_preds.csv", y_test_ae)


# Concatenate predictions and labels for 3-D visualization

# In[16]:


preds = pd.read_csv(f"preds/cnn_ae_preds.csv")
preds["type"] = 0
truth = pd.read_csv(f"truth.csv")
truth.columns=["_", "x", "y", "z"]
truth = truth[["x", "y", "z"]]
truth["type"] = 1
concat = pd.concat([preds,truth], ignore_index=False)
#concat


# In[17]:


from plotly.graph_objs import *
init_notebook_mode()
preds = pd.read_csv(f"preds/cnn_ae_preds.csv")

trace0 = Scatter3d(x=concat["x"], y=concat["y"], z=concat["z"], mode='markers',
                       marker=dict(
                           size=2,
                           color=concat['type'],
                           colorscale=[[0, "red"], [1, "blue"]],
                           symbol='circle',
                           opacity=0.5
                       )
                       )
data = [trace0]
layout = Layout(showlegend=False, height=600, width=600)
fig = dict(data=data, layout=layout)
plot_title = "3D plot by plotly"
plotly.offline.plot(fig, filename='3d_plot.html'.format(plot_title))
iplot(fig)


# In[ ]:




