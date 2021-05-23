# import libraries
import pandas as pd
import numpy as np
import random
import math
import scipy.io
from sklearn.model_selection import train_test_split
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
from tensorflow.keras.models import load_model
import os
font_size = 18

# read two raw datasets
mat1 = scipy.io.loadmat('dataset1.mat')
signal = mat1["transmitSignal"][0][0]
theta = mat1["pilotMatrix4N"].T
y = mat1["receivedSignal4N"].T
x = mat1["transmitSignal"].T
print("Loading dataset 1...")
print(f"y shape: {y.shape}, theta shape: {theta.shape}")

print("Loading dataset 2")
mat2 = scipy.io.loadmat('dataset2.mat')
y2 = mat2["receivedSignal"]
thetas2 = mat2["pilotMatrix"]
print(f"y shape: {y2.shape}, theta shape: {thetas2.shape}")

h_est = np.load('h_est.npy')
squares = np.square(np.absolute(h_est))

# exclude outliers in dataset1
thld = 25
ss = [(s1 * 1e12 > thld).sum() for s1 in squares]
re = [s for s in ss if s > 0]
len(re), squares.shape

_square_big = squares * 1e12 # scale signal magnitudes
square_normal = []
normal_indices = []
for idx, _square in enumerate(_square_big):
  if (_square > thld).sum() == 0:
    square_normal.append(_square)
    normal_indices.append(idx)

square_normal = np.array(square_normal)
print(square_normal.shape, _square_big.shape)

# train test split
theta_normal = theta[normal_indices]
x_train, x_test , y_train , y_test = train_test_split(theta_normal, square_normal, test_size=0.2, random_state=42)
print("train test split done")
print(f"training size: {y_train.shape}")

# phase 1: train a base model on dataset 1
opt = Adadelta(learning_rate=0.4, rho=0.95, epsilon=1e-07, name="Adadelta")
opt_adamax = Adamax(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adamax")
#opt = Adam(2)
opt_nadam = Nadam(learning_rate=0.01, beta_1=0.5, beta_2=0.999, epsilon=1e-07, name="Nadam")
opt_RMSprop = RMSprop(learning_rate=0.01,rho=0.9,momentum=0.0,epsilon=1e-07,centered=False,name="RMSprop",)
opt_adam = Adam(learning_rate=0.01,beta_1=0.5,beta_2=0.999,epsilon=1e-07,amsgrad=False,name="Adam")
opt_sgd = SGD(learning_rate=0.05, momentum=0.0, nesterov=False, name="SGD")
opt_ftrl = Ftrl(
    learning_rate=0.01,
    learning_rate_power=-0.5,
    initial_accumulator_value=0.1,
    l1_regularization_strength=0.0,
    l2_regularization_strength=0.0,
    name="Ftrl",
)
early_stopper = EarlyStopping(
    monitor="val_loss",
    mode="auto",
    patience=15
)
cp = ModelCheckpoint(filepath=f"model/model2", save_weights_only=False, monitor='val_loss',
                     mode='min',save_best_only=True)
kr = regularizers.l1_l2(l1=1e-5, l2=1e-4)
def dense_model(input_shape, output_shape, opt=opt, activation=ReLU, rgl=kr, final_act='softplus'):
    model = Sequential()
    model.add(Dense(256, 
                    input_shape=input_shape,
                    kernel_regularizer=rgl
            ))
    model.add(activation())
    model.add(Dense(128, 
                    kernel_regularizer=rgl
                   ))
    model.add(activation())
    model.add(Dense(64,
                    kernel_regularizer=rgl
                   ))
    model.add(activation())
    model.add(Dense(32, 
                    kernel_regularizer=rgl
                   ))
    model.add(activation())
    if final_act is not None:
        model.add(Dense(output_shape, activation=final_act))
    else:
        model.add(Dense(output_shape))
    #model.add(ReLU())
    model.compile(loss="mse", optimizer=opt)
    return model

model = dense_model(x_train.shape[1:], y_train.shape[1],
                    opt=opt_adam, final_act="softplus")
hist = model.fit(x_train, y_train , epochs=10, validation_data=(x_test, y_test),
                  callbacks=[early_stopper, cp], batch_size=16, verbose=1)

# save trained model to disk
if not os.path.isdir("model"):
    os.mkdir("model")
model.save("model/model2")

# phase 2. Fine training 50 models for 50 users
def extract_user(_no):
    _user = []
    for i in range(y2.shape[1]):
        _user.append([])
    for aa, _k in enumerate(y2):
        for idx, ll in enumerate(_k):
            #print(f"cp2:_theta[_no]:{np.array(_theta[_no]).shape}")
            _user[idx].append(ll[_no])
    tt = np.array(_user)
    return tt

def remove_outliers(array, thld=25):
  square_normal = []
  normal_indices = []
  for idx, _square in enumerate(array):
      if (_square > thld).sum() == 0:
          square_normal.append(_square)
          normal_indices.append(idx)

  square_normal = np.array(square_normal)
  return normal_indices, square_normal

def create_diag(dim, val):
    a = np.zeros((dim, dim))
    np.fill_diagonal(a, val)
    return a

def gen_early_stopper(patience=10):
    return EarlyStopping(monitor="val_loss",mode="auto",patience=patience)
def gen_model_checkpoint(user_no):
    return ModelCheckpoint(filepath=f"model/phase2/user{user_no}", save_weights_only=False, monitor='val_loss',
                           mode='min',save_best_only=True)

def further_train_user(_no, base_model=model, epochs=20, opt=opt_RMSprop, verbose=0, patience=10, thld_ratio=99, early_stop=True):
    _y = extract_user(_no)
    _x_diag = create_diag(_y.shape[0], signal)
    _x_inv = np.linalg.inv(_x_diag)
    _h_est = np.matmul(_x_inv, _y)
    print(f"Data extracted for user {_no}")
    _square = np.square(np.absolute(_h_est))
    _square_big = _square * 1e12
    thld = np.percentile(_square_big.reshape(-1), thld_ratio)
    _indices, _square_normal = remove_outliers(_square_big, thld)
    thetas2_normal = thetas2[_indices]
    
    # train test split
    _x2_train, _x2_test , _y2_train , _y2_test = train_test_split(thetas2, _square, test_size=0.2, random_state=42)
    _y2_train, _y2_test = _y2_train * 1e12, _y2_test * 1e12
    
    # copy model
    _model_copy = clone_model(base_model)
    _model_copy.set_weights(base_model.get_weights())
    _model_copy.compile(loss="mse", optimizer=opt)
    callbacks = [gen_model_checkpoint(_no)]
    if early_stop is True:
      callbacks.append(gen_early_stopper(patience))
    _hist = _model_copy.fit(_x2_train, _y2_train , epochs=epochs, validation_data=(_x2_test, _y2_test),
                  callbacks=callbacks,
                  batch_size=16, verbose=verbose)
    print(f"training finished for user {_no}, {len(_hist.history['loss'])} epochs, best loss: {min(_hist.history['val_loss'])}")
    return _model_copy, _hist

do the training for 50 users, and save models to disk
models, hists = [], []
for i in range(50):
    _model, _hist = further_train_user(i, epochs=60)
    models.append(_model)
    _model.save(f"model/phase2/user{i}")
    hists.append(_hist)

# print("loading models...")
# models = []
# for i in range(50):
#     _model = load_model(f"model/phase2/user{i}")
#     models.append(_model)

# phase 3: customized back propagation
def log2(x):
    numerator = K.log(x)
    denominator = K.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator

BN0 = 3.06725096231643e-13
def customized_loss(y_true, y_pred): # implement the customized loss to be the final evaluation formula
    B, K_subc, M, P = 10e6, 500, 20, 1
    _coef = B / (K_subc + M - 1)
    return -(_coef * K.sum(log2(y_pred * P / BN0 + 1)))

# construct empty initializations for BP training/testing set
cl_train_x = np.full([13107, 1], 1)
cl_train_y = np.full((13107, 500), 1)
cl_test_x = np.full([3277, 1], 1)
cl_test_y = np.full((3277, 500), 1)

# customize a new NN structure for BP:
# 1. add a 4096-node layer before
# 2. use a customized loss
def retrain_optimize_theta(input_model, epochs=10, activation=ReLU, optimizer=Adam(2), verbose=0, theta=None):
    _model = Sequential()
    if theta is None:
        initializer = initializers.Constant(value=0)
    else:
        initializer = lambda shape, dtype: K.variable(theta2)
    _model.add(Dense(4096,
                     input_shape=(1,),
                     kernel_initializer=initializer,
                     kernel_constraint=MinMaxNorm(min_value=-1, max_value=1),
                     use_bias=False
                    )
              )

    _model.add(input_model)
    _model.layers[1].trainable = False
    _model.compile(loss=customized_loss, optimizer=optimizer)
    _hist = _model.fit(cl_train_x, cl_train_y, epochs=epochs, validation_data=(cl_test_x, cl_test_y),
                  callbacks=[early_stopper],
                  batch_size=16, verbose=verbose) # try larger batch sizes

    print(f"Done, {len(_hist.history['loss'])} epochs, best loss: {min(_hist.history['val_loss'])}")
    return _model, _hist

do the BP for 50 users
models_bp, hists_bp = [], []
for i in range(50):
    print(f"optimizing user {i}")
    #_best_theta = best_thetas[i].reshape(1, 4096)
    _model, _hist = retrain_optimize_theta(models[i], epochs=100, optimizer=opt_adamax, theta=None)
    print(f"{len(_hist.history['loss'])} epochs trained")
    models_bp.append(_model)
    hists_bp.append(_hist)

# save BP models to disk
for idx, _model in enumerate(models_bp):
  _model.save(f"model/bp/user{idx}")


# load saved phase2 and bp models
models = []
for i in range(50):
    _model = load_model(f"model/phase2/user{i}")
    models.append(_model)

model_bp = []
for i in range(50):
    _model = load_model(f"model/bp/user{i}", custom_objects={'customized_loss': customized_loss})
    model_bp.append(_model)

# get the optimized theta for a specific user 
def get_theta_for_user(_user_no):
	return model_bp[_user_no].layers[0].get_weights()[0][0]


def thresholding(_list): # threshold model weights into -1 or 1
    results = []
    for _v in _list:
        if _v < 0:
            results.append(-1)
        else:
            results.append(1)
    return np.array(results)

B, K_subc, M, P = 10e6, 500, 20, 1
_coef = B / (K_subc + M - 1)

def cal_r_from_theta(_theta, models=None, model=None, user_no=0):
    if model is not None:
        y_pred = model.predict(np.array([_theta]))
    else:
        y_pred = models[user_no].predict(np.array([_theta]))
    y_pred_real = y_pred / 1e12
    return -(_coef * np.sum(np.log2(y_pred_real * P / BN0 + 1)))

# generate uniform random theta
def generate_random_theta():
    _result = []
    for i in range(4096):
        if random.random() < 0.5:
            _result.append(-1)
        else:
            _result.append(1)
    return np.array(_result)

g = generate_random_theta()
theta_all_neg1 = np.full(4096, -1)
theta_all1 = np.full(4096, 1)

# calculate final rates by different theta solutions
randoms, all_neg1, all1 = [], [], []
latest_0509, latest_0506 = [], []
for i in range(50):

    _theta_latest_0509 = model_bp[i].layers[0].get_weights()[0][0]
    latest_0509.append(-cal_r_from_theta(_theta_latest_0509, models=models, user_no=i))
    
    theta_random = generate_random_theta()
    randoms.append(-cal_r_from_theta(theta_random, models, user_no=i))
    all_neg1.append(-cal_r_from_theta(theta_all_neg1, models, user_no=i))
    all1.append(-cal_r_from_theta(theta_all1, models, user_no=i))

print(f"the final rates by our algorithm: {latest_0509}")

