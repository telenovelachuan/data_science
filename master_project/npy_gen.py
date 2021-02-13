import pandas as pd
import numpy as np
from modAL.models import ActiveLearner
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Activation, AveragePooling2D, Conv2DTranspose, ZeroPadding2D
from tensorflow.keras.layers import Dense , Dropout, Conv2D ,Flatten, Conv1D, BatchNormalization
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

from matplotlib import pyplot as plt
import seaborn as sns
import copy
import math

font_size = 18

# load numpy raw data
with open('all_ae.npy', 'rb') as f:
    h2_ae = np.load(f)
    Pos = np.load(f)

h2_ae_shape = h2_ae.shape
h2_ae_1dcnn = h2_ae.reshape(h2_ae_shape[0], h2_ae_shape[1], -1)
h2_ae.shape, Pos.shape

# cluster the dataset for classification
pos_cluster = pd.DataFrame(Pos, columns=["x", "y", "z"])
kmeans = KMeans(n_clusters=5, random_state=42).fit(pos_cluster)
pos_cluster["cluster"] = kmeans.labels_
pos_cluster["cluster"] = pos_cluster["cluster"].replace({0: "cluster 1", 1: "cluster 2", 2: "cluster 3", 3: "cluster 4", 4: "cluster 5"})
pos_cluster["is_c1"] = 0
pos_cluster.loc[pos_cluster.cluster == "cluster 1", 'is_c1'] = 1
pos_cluster["is_c2"] = 0
pos_cluster.loc[pos_cluster.cluster == "cluster 2", 'is_c2'] = 1
pos_cluster["is_c3"] = 0
pos_cluster.loc[pos_cluster.cluster == "cluster 3", 'is_c3'] = 1
pos_cluster["is_c4"] = 0
pos_cluster.loc[pos_cluster.cluster == "cluster 4", 'is_c4'] = 1
pos_cluster["is_c5"] = 0
pos_cluster.loc[pos_cluster.cluster == "cluster 5", 'is_c5'] = 1

print("train-test splitting ae data...")
x_train_ae, x_test_ae, y_train_ae, y_test_ae = train_test_split(h2_ae, pos_cluster, test_size=0.1, random_state=42)
x_train_ae_cnn, x_test_ae_cnn, y_train, y_test = train_test_split(h2_ae_1dcnn, pos_cluster, test_size=0.1, random_state=42)

#y_train_ae_cnn = y_train[["x", "y", "z"]].values
y_test_ae_cnn = y_test[["x", "y", "z"]].values
print("splitting training data and active learning pool...")
train_x_al, pool_x_al, train_y_al, pool_y_al = train_test_split(x_train_ae_cnn, y_train, test_size=0.8, random_state=42)

y_train_cluster = y_train[["is_c1", "is_c2", "is_c3"]].values
y_test_cluster = y_test[["is_c1", "is_c2", "is_c3"]].values
train_x_cluster, pool_x_cluster, train_y_cluster, pool_y_cluster = train_test_split(x_train_ae_cnn, y_train_cluster, test_size=0.8, random_state=42)

def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

print("saving distance(pool dots, train dots of same cluster) into dist_dict_pretrain...")
dist_dict = []
for i in range(len(pool_x_al)):
    if i % 100 == 0:
        print(f"{i} / {len(pool_x_al)}")
    _current_cluster = pool_y_al.iloc[i]["cluster"]
    s = sum([distance(pool_x_al[i], train_x_al[j]) for j in range(len(train_x_al)) if _current_cluster == train_y_al.iloc[j]["cluster"]])
    if i == 0:
        print(f"at zero, pool_x_al[i]:{pool_x_al[i]}, sum:{s}")
        print(f"result 0: {s}")
    dist_dict.append(s)
dist_dict = np.array(dist_dict)
with open('dist_dict_pretrain.npy', 'wb') as f:
    np.save(f, dist_dict)
print("dist_dict_pretrain saved!")


print("saving dist_dict...")
dist_dict = []
for i in range(len(pool_x_al)):
    if i % 100 == 0:
        print(f"{i} / {len(pool_x_al)}")
    _current_cluster = pool_y_al.iloc[i]["cluster"]
    _row = []
    for j in range(len(pool_x_al)):
        _row.append(distance(pool_x_al[i], pool_x_al[j]))
    dist_dict.append(_row)
dist_dict = np.array(dist_dict)
with open('dist_dict.npy', 'wb') as f:
    np.save(f, dist_dict)
print("dist_dict saved!")

