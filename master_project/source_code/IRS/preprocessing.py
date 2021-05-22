# import libraries
import pandas as pd
import numpy as np
import random
import math
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from matplotlib import pyplot as plt
import seaborn as sns

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential, load_model, clone_model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import *
from tensorflow.keras import activations
from tensorflow.keras.constraints import *
from tensorflow.keras import regularizers

import os
import warnings
warnings.filterwarnings('ignore')
font_size = 18

# read dataset 1
mat1 = scipy.io.loadmat('dataset1.mat')
signal = mat1["transmitSignal"][0][0]
theta = mat1["pilotMatrix4N"].T
y = mat1["receivedSignal4N"].T
x = mat1["transmitSignal"].T
print("Loading dataset 1...")
print(f"y shape: {y.shape}, theta shape: {theta.shape}")

# read dataset 2
print("Loading dataset 2")
mat2 = scipy.io.loadmat('dataset2.mat')
y2 = mat2["receivedSignal"]
thetas2 = mat2["pilotMatrix"]
print(f"y shape: {y2.shape}, theta shape: {thetas2.shape}")

# some calculation to get noised estimation of h_theta
def create_diag(dim, val):
    a = np.zeros((dim, dim))
    np.fill_diagonal(a, val)
    return a

x_diag = create_diag(y.shape[0], signal)
x_inv = np.linalg.inv(x_diag)
h_est = np.matmul(x_inv, y)
# save to disk
with open('h_est.npy', 'wb') as f:
    np.save(f, h_est)
print(f"h_est shape: {h_est.shape}")

# calculate the squared magnitude of each signal
squares = np.square(np.absolute(h_est))

# denoising using autoencoder
# adjust the structure of dataset2 for denoising
y2_reshaped = [] # 4096 x 500 for each user
os.mkdir("y2_reshape")
for user_idx in range(50):
  _user_reshape = []
  for i in range(4096):
    _user_reshape.append([])

  for _k in y2:
    for idx, _theta in enumerate(_k):
      _user_reshape[idx].append(_theta[user_idx])
  
  # save to disk
  with open(f'y2_reshape/user{user_idx}.npy', 'wb') as f:
    np.save(f, _user_reshape)
  print(f"user{user_idx} reshaped")

# add noise onto 50 users data
noise = (y[0] - y[8192]) / math.sqrt(2)
noise_real, noise_imag = np.real(noise), np.imag(noise)
noise_var_real, noise_var_imag = np.var(noise_real), np.var(noise_imag)
std_real, std_imag = math.sqrt(noise_var_real), math.sqrt(noise_var_imag)

real_noise = np.random.normal(loc=0.0, scale=std_real, size=(4096,500))
imag_noise = np.random.normal(loc=0.0, scale=std_imag, size=(4096,500))
complex_noise = real_noise + imag_noise * 1j

os.mkdir("y2_noised")
for i in range(50):
  _u = np.load(f"y2_reshape/user{i}.npy")
  _u_noised = _u + complex_noise
  with open(f'y2_noised/user{i}.npy', 'wb') as f:
    np.save(f, _u_noised)
  print(f"user{i} noise added")

# building autoencoder
def ae_model(input_shape, opt=Adam(1e-3), activation=ReLU, init="lecun_uniform"):
    model = Sequential()
    #model.add(Flatten())
    model.add(Dense(256, kernel_initializer=init))
    model.add(activation())
    #model.add(BatchNormalization())
    #model.add(Dropout(rate=dropout_rate))
    model.add(Dense(128, kernel_initializer=init))
    model.add(activation())
    #model.add(BatchNormalization())
    #model.add(Dropout(rate=dropout_rate))
    model.add(Dense(64, kernel_initializer=init))
    model.add(activation())
    #model.add(BatchNormalization())
    
    model.add(Dense(128, kernel_initializer=init))
    model.add(activation())
    #model.add(BatchNormalization())
    
    model.add(Dense(256, kernel_initializer=init))
    model.add(activation())
    #model.add(BatchNormalization())

    model.add(Dense(real_y2_train.shape[1]))
    model.compile(loss="mse", optimizer=opt) 
    return model

# denoising
opt = Adamax(learning_rate=0.01, beta_1=0.5, beta_2=0.999, epsilon=1e-07, name="Adamax")
opt_adam = Adam(0.1)
opt_RMSprop = RMSprop(learning_rate=0.01,rho=0.9,momentum=0.0,epsilon=1e-07,centered=False,name="RMSprop")
opt_adagrade = Adagrad(learning_rate=0.01,initial_accumulator_value=0.1,epsilon=1e-07,name="Adagrad")
opt_nadam = Nadam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam")
opt_ftrl= Ftrl(
      learning_rate=0.01,
      learning_rate_power=-0.5,
      initial_accumulator_value=0.1,
      l1_regularization_strength=0.0,
      l2_regularization_strength=0.0,
      name="Ftrl",
      l2_shrinkage_regularization_strength=0.0,
)
# for saving results to disk
os.mkdir("y2_denoised_imag")
os.mkdir("y2_denoised_real")
def denoise_user(user_no):

  u0_noised = np.load(f"y2_noised/user{user_no}.npy")
  u0_orig = np.load(f"y2_reshape/user{user_no}.npy")
  # train test split
  u0 = np.load("y2_noised/user0.npy")
  real_x2_train, real_x2_test , real_y2_train , real_y2_test = train_test_split(np.real(u0_noised), np.real(u0_orig), test_size=0.2, random_state=42)
  real_x2_train = real_x2_train * 1e10
  real_x2_test = real_x2_test * 1e10
  real_y2_train = real_y2_train * 1e10
  real_y2_test = real_y2_test * 1e10

  imag_x2_train, imag_x2_test , imag_y2_train , imag_y2_test = train_test_split(np.imag(u0_noised), np.imag(u0_orig), test_size=0.2, random_state=42)
  imag_x2_train = imag_x2_train * 1e10
  imag_x2_test = imag_x2_test * 1e10
  imag_y2_train = imag_y2_train * 1e10
  imag_y2_test = imag_y2_test * 1e10

  # construct autoencoder
  early_stopper = EarlyStopping(
      monitor="val_loss",
      mode="auto",
      patience=5
  )
  # real part
  u0_real = ae_model(real_x2_train.shape[1:], 
                #opt=Adam(5e-2),
                opt=opt_adagrade
              )
  u0_hist_real = u0_real.fit(real_x2_train, real_y2_train , epochs=300, validation_data=(real_x2_test, real_y2_test),
                    callbacks=[early_stopper],
                    batch_size=16, verbose=0)

  denoised_real = u0_real.predict(np.real(u0_orig) * 1e10) / 1e10
  with open(f'y2_denoised_real/user{i}.npy', 'wb') as f:
    np.save(f, denoised_real)
  print(f"user{i} real done: {u0_hist_real.history['val_loss'][-1]}")

  # imag part
  u0_imag = ae_model(imag_x2_train.shape[1:], 
                #opt=Adam(5e-2),
                opt=opt_adagrade
              )
  u0_hist_imag = u0_real.fit(imag_x2_train, imag_y2_train , epochs=300, validation_data=(imag_x2_test, imag_y2_test),
                    callbacks=[early_stopper],
                    batch_size=16, verbose=0)
  denoised_imag = u0_imag.predict(np.imag(u0_orig) * 1e10) / 1e10
  with open(f'y2_denoised_imag/user{i}.npy', 'wb') as f:
    np.save(f, denoised_imag)
  print(f"user{i} imag done: {u0_hist_imag.history['val_loss'][-1]}")

for i in range(50):
  denoise_user(i)

