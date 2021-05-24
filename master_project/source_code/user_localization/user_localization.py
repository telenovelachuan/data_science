# import libraries
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
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Activation, AveragePooling2D, Conv2DTranspose, ZeroPadding2D
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, Conv1D, BatchNormalization
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from modAL.models import ActiveLearner
from modAL.uncertainty import entropy_sampling, margin_sampling, uncertainty_sampling
import copy

CTW_labelled = "/home/c693s270/"
CLUSTER_NUM = 5
pretrain_npy_name = f'dist_dict_pretrain_{CLUSTER_NUM}cluster.npy'


# load data from pre-saved file
print("Loading all training data...")
with open('all_ae.npy', 'rb') as f:
    h2_ae = np.load(f)
    Pos = np.load(f)

# cluster the dataset into 5 clusters, which is later used by the location based query strategy
pos_cluster = pd.DataFrame(Pos, columns=["x", "y", "z"])
kmeans = KMeans(n_clusters=CLUSTER_NUM, random_state=42).fit(pos_cluster)
pos_cluster["cluster"] = kmeans.labels_
replace_dict = {}
for i in range(CLUSTER_NUM):
    replace_dict[i] = f"cluster {i + 1}"
pos_cluster["cluster"] = pos_cluster["cluster"].replace(replace_dict)
for i in range(CLUSTER_NUM):
    pos_cluster[f"is_c{i + 1}"] = 0
    pos_cluster.loc[pos_cluster.cluster == f"cluster {i}", f'is_c{i}'] = 1


# construct the base learner regression model
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

# the base learner classification model that outputs probability
def cnn_model_clf(input_shape, opt=Adam(1e-3), dropout_rate=0.2):
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
    model.add(Dense(CLUSTER_NUM, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# reshape data to be inputted into our model
h2_ae_shape = h2_ae.shape
h2_ae_1dcnn = h2_ae.reshape(h2_ae_shape[0], h2_ae_shape[1], -1)

# train-test split into 3 parts:
# 1. 10% test set
# 2. 18% training set
# 3. 72% active learning query pool
print("train-test splitting ae data...")
x_train_ae, x_test_ae, y_train_ae, y_test_ae = train_test_split(h2_ae, pos_cluster, test_size=0.1, random_state=42)
x_train_ae_cnn, x_test_ae_cnn, y_train, y_test = train_test_split(h2_ae_1dcnn, pos_cluster, test_size=0.1, random_state=42)
y_test_ae_cnn = y_test[["x", "y", "z"]].values

print("splitting training data and active learning pool...")
train_x_al, pool_x_al, train_y_al, pool_y_al = train_test_split(x_train_ae_cnn, y_train, test_size=0.8, random_state=42)

cluster_bool_cols = [f"is_c{i + 1}" for i in range(CLUSTER_NUM)]
y_train_cluster = y_train[cluster_bool_cols].values
y_test_cluster = y_test[cluster_bool_cols].values
train_x_cluster, pool_x_cluster, train_y_cluster, pool_y_cluster = train_test_split(x_train_ae_cnn, y_train_cluster, test_size=0.8, random_state=42)

# pre-calculate the distance between a training sample and all other samples in a same cluster, and save to disk 
print("saving distance(pool dots, train dots of same cluster) into dist_dict_pretrain...")
def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

if not os.path.isfile(pretrain_npy_name):
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
    with open(pretrain_npy_name, 'wb') as f:
        np.save(f, dist_dict)
    print("dist_dict_pretrain saved!")

# create 2 base learner models:
# 1. a classification model for our location based QS
# 2. a regression model for all other QS
print("constructing 1D-CNN models...")
lr = 5e-3
epochs = 100
decay_rate = lr / 50
model_ae = cnn_model1(train_x_al.shape[1:], opt=Adam(5e-3, decay=decay_rate))
model_clf = cnn_model_clf(train_x_cluster.shape[1:], opt=Adam(5e-3, decay=decay_rate))

def get_prediction_precision(regressor, x_test, y_test):
    y_pred = regressor.predict(x_test)
    mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    #print(f"current MSE: {mse}")
    return mse

# implementation for the random sampling QS, since it's not implemented by the modAL lib
def random_sampling(classifier, X_pool):
    n_samples = len(X_pool)
    query_idx = np.random.choice(range(n_samples))
    return [query_idx], X_pool[query_idx]

def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

# implementation for the proposed location based QS
xs, ys_reg, ys_clf = [], [], []
xs_idx = []
dist_dict = np.load('dist_dict.npy')
dist_dict_pretrain = np.load(pretrain_npy_name)

def location_based_sampling(classifier, X_pool):
    overall_distances = [sum([dist_dict[_chosen_idx][cur_idx] for _chosen_idx in xs_idx]) + dist_dict_pretrain[cur_idx] for cur_idx, x in enumerate(X_pool)]
    overall_distances_np = np.array(overall_distances)
    maxes = classifier.predict_proba(X_pool).max(axis=1)
    uncertainties = np.full(len(overall_distances), 1) - maxes
    product = overall_distances_np * uncertainties
    #query_idx = list(product).index(max(product))
    query_idx = -1
    product_sorted = sorted(product, reverse=True)
    for _prod in product_sorted:
        _query_idx = list(product).index(_prod)
        if _query_idx not in xs_idx:
            query_idx = _query_idx
            break
    return [query_idx], [X_pool[query_idx]]


# train both base models on the 16% training set
train_y_al = train_y_al[["x", "y", "z"]].values
print(f"train_x_al:{train_x_al.shape}, train_y_al:{train_y_al.shape}")
model_ae.fit(train_x_al, train_y_al, epochs=1, validation_data=(
    x_test_ae_cnn, y_test_ae_cnn), verbose=1)
model_clf.fit(train_x_cluster, train_y_cluster, epochs=5, validation_data=(
    x_test_ae_cnn, y_test_cluster), verbose=1)
print(
    f"Initial round of training finished, MSE:{get_prediction_precision(model_ae, x_test_ae_cnn, y_test_ae_cnn)}")
print("Begin active learning process...")

# below is the main process of active learning.
# the mode variable can be changed to run different query strategies. Valid options:
# 1. uncertainty. For uncertainty sampling
# 2. random. For random sampling
# 3. margin. For margin sampling
# 4. entropy. For entropy sampling
# 5. lb. For location based sampling

n_queries = 1700
mode = "uncertainty"
#queries_num = np.linspace(100, 3500, 35)
mse_dict = {}
all_chosen_samples_reg = []
all_chosen_samples_clf = []

mode_dict = {
    "random": random_sampling,
    "uncertainty": uncertainty_sampling,
    "margin": margin_sampling,
    "entropy": entropy_sampling,
    "lb": location_based_sampling
}
print("Initialize regression active learner...")
regressor = ActiveLearner(
    estimator=model_ae,
    query_strategy=mode_dict[mode],
    X_training=train_x_al, y_training=train_y_al
)
print("Initialize classification active learner...")
classifier = ActiveLearner(
    estimator=model_clf,
    query_strategy=mode_dict[mode],
    X_training=train_x_al, y_training=train_y_cluster
)
print(f"active learning for {n_queries} epochs")

i = 0
pool_x_shape = pool_x_al.shape
pool_y_al = pool_y_al[["x", "y", "z"]].values
best_mse = 999999999
best_iteration = -1
while i <= n_queries:
    pool_x_al_unchosen = np.array([t for idx,t in enumerate(list(pool_x_al)) if idx not in xs_idx])
    if mode not in ["lb", "random"]:
        query_idx, query_instance = classifier.query(pool_x_al_unchosen)
    else:
        query_idx, query_instance = regressor.query(pool_x_al_unchosen)
    query_idx = -1
    for idx,x in enumerate(pool_x_al):
        if np.array_equal(x, query_instance[0]):
            query_idx = idx
            break
    _x, _y_reg, _y_clf = pool_x_al[query_idx], pool_y_al[query_idx], pool_y_cluster[query_idx]
    all_chosen_samples_reg.append(_y_reg)
    all_chosen_samples_clf.append(_y_clf)

    xs.append(_x)
    ys_reg.append(_y_reg)
    ys_clf.append(_y_clf)
    if not type(query_idx) == int:
        query_idx = query_idx[0]
    xs_idx.append(query_idx)

    # for every 50 iterations, teach the base learner the newly selected samples
    if i % 50 == 0 and i > 0:
        # batch mode AL teach
        regressor_copy = copy.copy(regressor)
        classifier_copy = copy.copy(classifier)

        xs_backup = xs.copy()
        ys_reg_backup = ys_reg.copy()
        ys_clf_backup = ys_clf.copy()
        xs_idx_backup = xs_idx.copy()
        xs = np.array(xs)
        ys_reg = np.array(ys_reg)
        ys_clf = np.array(ys_clf)
        x_shape, y_shape = xs.shape, ys_reg.shape
        xs = xs.reshape(x_shape[0], -1, x_shape[-1])
        ys_reg = ys_reg.reshape(y_shape[0], -1)
        ys_clf = ys_clf.reshape(y_shape[0], -1)
        print(f"xs:{xs.shape}, ys:{ys_reg.shape}")
        regressor.teach(xs, ys_reg, only_new=False)
        if mode not in ["lb", "random"]:
            classifier.teach(xs, ys_clf, only_new=False)

        # evaluate the newly taught base model, and save best metrics and predictions
        mse = get_prediction_precision(regressor, x_test_ae_cnn, y_test_ae_cnn)
        if mse < best_mse:
            print("best mse ever! saving model...")
            model_ae.save(f"models/model_ae_al2_{mode}_best.h5")
            print("saving predictions...")
            preds = model_ae.predict(x_test_ae_cnn)
            result = pd.DataFrame(preds, columns=["x", "y", "z"])
            print(f"{len(result)} rows")
            result.to_csv(f"preds/cnn_ae_al2_{mode}_best.csv")
            best_mse = mse
            best_iteration = i


        mse_dict[i] = mse
        print(f"Evaluating model at {i}th query...mse:{mse}")
        xs, ys_reg, ys_clf, xs_idx = [], [], [], []
    i += 1

# save the final base model
print("saving model...")
model_ae.save(f"models/model_ae_al3_{mode}_all.h5")

# make final predictions
print("making predictions & saving results...")
preds = model_ae.predict(x_test_ae_cnn)
result = pd.DataFrame(preds, columns=["x", "y", "z"])
print(f"{len(result)} rows")
result.to_csv(f"preds/cnn_ae_al3_{mode}_all_preds.csv")

# save the chosen samples
print("saving chosen samples")
df_chosen = pd.DataFrame(
    np.array(all_chosen_samples_reg).reshape(-1, 3), columns=["x", "y", "z"])
df_chosen.to_csv(f"chosen/cnn_ae_al3_{mode}_all_chosen.csv")
print(f"All done!! mode:{mode}")
print(f"mse_dict: {mse_dict}")
print(f"best iteration: {best_iteration}")

