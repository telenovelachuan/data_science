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

font_size = 18

# read two raw datasets
mat1 = scipy.io.loadmat(path + 'dataset1.mat')
signal = mat1["transmitSignal"][0][0]
theta = mat1["pilotMatrix4N"].T
y = mat1["receivedSignal4N"].T
x = mat1["transmitSignal"].T
print("Loading dataset 1...")
print(f"y shape: {y.shape}, theta shape: {theta.shape}")

print("Loading dataset 2")
mat2 = scipy.io.loadmat(path + 'dataset2.mat')
y2 = mat2["receivedSignal"]
thetas2 = mat2["pilotMatrix"]
print(f"y shape: {y2.shape}, theta shape: {thetas2.shape}")


