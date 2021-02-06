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


def get_data(data_file):

    f = h5py.File(data_file, 'r')
    H_Re = f['H_Re'][:]  # shape (sample size, 56, 924, 5)
    H_Im = f['H_Im'][:]  # shape (sample size, 56, 924, 5)
    SNR = f['SNR'][:]  # shape (sample size, 56, 5)
    Pos = f['Pos'][:]  # shape(sample size, 3)
    f.close()

    return H_Re, H_Im, SNR, Pos



# # load data from file_1.hdf4

# data_file = CTW_labelled + "file_" + str(1) + ".hdf5"
# H_Re, H_Im, SNR, Pos = get_data(data_file)
# print("H_Re is of shape {}".format(H_Re.shape))
# print("H_Im is of shape {}".format(H_Im.shape))
# print("SNR is of shape {}".format(SNR.shape))
# print("Pos is of shape {}".format(Pos.shape))

# # Euclidean norm
# H = np.sqrt(H_Re[:,:]**2 + H_Im[:,:]**2)

# # Extracting phase
# print("extracting phase...")
# arctan = np.divide(H_Re[:,:], H_Im[:,:])
# arctan[np.isnan(arctan)] = math.pi / 2
# #arctan.shape

# # append phase onto H
# print("appending phase...")
# h1 = np.empty([512, 56, 924, 10])
# for d1 in range(512):
#     for d2 in range(56):
#         for d3 in range(924):
#             h = H[d1][d2][d3]
#             phase = arctan[d1][d2][d3]
#             h1[d1][d2][d3] = np.vstack((h, phase)).reshape(-1)

# #h1.shape

# # append SNR onto h1
# print("appending SNR...")
# snr_rep = np.repeat(SNR, 2, axis=1).reshape(512, 56, -1)
# h2 = np.empty([512, 56, 925, 10])
# for d1 in range(512):
#     for d2 in range(56):
#         h = h1[d1][d2]
#         snr = snr_rep[d1][d2]
#         h2[d1][d2] = np.vstack((h, snr))

# #h2.shape

# # replace inifinity values
# h2[h2 == inf] = 0
# h2[h2 == -inf] = 0


# load data from pre-saved file
print("Loading all training data...")
with open('all_ae.npy', 'rb') as f:
    # 4979 * 56 * 925 * 10
    h2_ae = np.load(f)
    Pos = np.load(f)

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

# model for outputting probability
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
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# def data_gen(data):
#     for i in range(len(data)):
#         yield (data[i: i+1],data[i: i+1])

# def autoencoder():
#     model = Sequential()
#     model.add(Conv2D(128, (5, 5),input_shape=h2.shape[1:], activation='relu', padding='same'))
#     model.add(AveragePooling2D((1, 5)))
#     model.add(Conv2D(32, (3, 3),activation='relu', padding='same'))
#     model.add(AveragePooling2D((1, 5)))
#     model.add(Conv2D(32,(3,3),activation='linear', padding='same'))
#     model.add(Conv2DTranspose(32, (3,3), strides=(1, 1), padding='same', activation='relu'))
#     model.add(Conv2DTranspose(128,(3,3), strides=(1, 5), padding='same', activation='relu'))
#     #model.add(ZeroPadding2D(((0, 0), (2, 1))))
#     model.add(Conv2DTranspose(10, (3, 3) , strides=(1, 5), padding='same', activation='linear'))
#     model.compile(loss='mean_squared_error', optimizer=Adam(1e-3))
#     return model

# ae = autoencoder()
# # ae.summary()

# print("Try AutoEncoder. Train-test split for AutoEncoder...")
# data, data_v  = train_test_split(h2, test_size=0.5, random_state=54)
# data.shape

# print("training AutoEncoder...")
# for i in range(5):
#     ae.fit_generator(data_gen(data),validation_data=data_gen(data_v), epochs=1, steps_per_epoch=len(data),
#                      validation_steps=len(data_v))

# print("applying encoder on original data...")
# encoder = Model(ae.input, ae.layers[-5].output)
# h2_ae = encoder.predict(h2)
h2_ae_shape = h2_ae.shape
h2_ae_1dcnn = h2_ae.reshape(h2_ae_shape[0], h2_ae_shape[1], -1)


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


# print("train-test splitting ae data...")
# x_train_ae, x_test_ae, y_train_ae, y_test_ae = train_test_split(
#     h2_ae, Pos, test_size=0.1, random_state=42)
# x_train_ae_cnn, x_test_ae_cnn, y_train_ae_cnn, y_test_ae_cnn = train_test_split(
#     h2_ae_1dcnn, Pos, test_size=0.1, random_state=42)

# print("splitting training data and active learning pool...")
# train_x_al, pool_x_al, train_y_al, pool_y_al = train_test_split(
#     x_train_ae_cnn, y_train_ae_cnn, test_size=0.8, random_state=42)
# print(f"Test samples: {len(y_test_ae_cnn)}, pool size: {len(pool_y_al)}")

# print("train-test splitting ae data on clustered pos...")
# x_train_cluster, x_test_cluster, y_train_cluster, y_test_cluster = train_test_split(h2_ae_1dcnn, pos_cluster, test_size=0.1, random_state=42)
print("train-test splitting ae data...")
x_train_ae, x_test_ae, y_train_ae, y_test_ae = train_test_split(h2_ae, pos_cluster, test_size=0.1, random_state=42)
x_train_ae_cnn, x_test_ae_cnn, y_train, y_test = train_test_split(h2_ae_1dcnn, pos_cluster, test_size=0.1, random_state=42)

y_train_ae_cnn = y_train[["x", "y", "z"]].values
y_test_ae_cnn = y_test[["x", "y", "z"]].values
print("splitting training data and active learning pool...")
train_x_al, pool_x_al, train_y_al, pool_y_al = train_test_split(x_train_ae_cnn, y_train_ae_cnn, test_size=0.8, random_state=42)

y_train_cluster = y_train[["is_c1", "is_c2", "is_c3", "is_c4", "is_c5"]].values
y_test_cluster = y_test[["is_c1", "is_c2", "is_c3", "is_c4", "is_c5"]].values
train_x_cluster, pool_x_cluster, train_y_cluster, pool_y_cluster = train_test_split(x_train_ae_cnn, y_train_cluster, test_size=0.8, random_state=42)


print("constructing 1D-CNN models...")
lr = 5e-3
epochs = 50
decay_rate = lr / 50
model_ae = cnn_model1(train_x_al.shape[1:], opt=Adam(5e-3, decay=decay_rate))
model_clf = cnn_model_clf(train_x_cluster.shape[1:], opt=Adam(5e-3, decay=decay_rate))

def get_prediction_precision(regressor, x_test, y_test):
    y_pred = regressor.predict(x_test)
    mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    #print(f"current MSE: {mse}")
    return mse


def random_sampling(classifier, X_pool):
    n_samples = len(X_pool)
    query_idx = np.random.choice(range(n_samples))
    return [query_idx], X_pool[query_idx]

def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

xs, ys_reg, ys_clf = [], [], []
xs_idx = []
dist_dict = np.load('dist_dict.npy')
dist_dict_pretrain = np.load('dist_dict_pretrain.npy')
def location_based_sampling(classifier, X_pool):
    # add up pretrain distances with newly-chosen distances
    overall_distances = [sum([dist_dict[_chosen_idx][cur_idx] for _chosen_idx in xs_idx]) + dist_dict_pretrain[cur_idx] for cur_idx, x in enumerate(X_pool)]
    #distances = [sum([dist_dict[chosen_idx][cur_idx] for chosen_idx in xs_idx]) for cur_idx, x in enumerate(X_pool)]
    query_idx = overall_distances.index(max(overall_distances))
    return [query_idx], X_pool[query_idx]

def location_based_sampling2(classifier, X_pool):
    overall_distances = [sum([dist_dict[_chosen_idx][cur_idx] for _chosen_idx in xs_idx]) + dist_dict_pretrain[cur_idx] for cur_idx, x in enumerate(X_pool)]
    overall_distances_np = np.array(overall_distances)
    maxes = classifier.predict_proba(X_pool).max(axis=1)
    uncertainties = np.full(len(overall_distances), 1) - maxes
    product = overall_distances_np * uncertainties
    query_idx = list(product).index(max(product))
    return [query_idx], X_pool[query_idx]


# 1. Fit on original training data
print(f"train_x_al:{train_x_al.shape}, train_y_al:{train_y_al.shape}")
model_ae.fit(train_x_al, train_y_al, epochs=1, validation_data=(
    x_test_ae_cnn, y_test_ae_cnn), verbose=1)
model_clf.fit(train_x_cluster, train_y_cluster, epochs=1, validation_data=(
    x_test_ae_cnn, y_test_cluster), verbose=1)
print(
    f"Initial round of training finished, MSE:{get_prediction_precision(model_ae, x_test_ae_cnn, y_test_ae_cnn)}")
print("Begin active learning process...")

# print("saving dist_dict...")
# dist_dict = []
# for i in range(len(pool_x_al)):
#     if i % 100 == 0:
#         print(f"{i} / {len(pool_x_al)}")
#     _row = []
#     for j in range(len(pool_x_al)):
#         _row.append(distance(pool_x_al[i], pool_x_al[j]))
#     dist_dict.append(_row)
# dist_dict = np.array(dist_dict)
# with open('dist_dict.npy', 'wb') as f:
#     np.save(f, dist_dict)
# print("dist_dict saved!")

# print("saving dist_dict_pretrain...")
# dist_dict = []
# for i in range(len(pool_x_al)):
#     if i % 100 == 0:
#         print(f"{i} / {len(pool_x_al)}")
#     s = sum([distance(pool_x_al[i], train_x_al[j]) for j in range(len(train_x_al))])
#     if i == 0:
#         print(f"at zero, pool_x_al[i]:{pool_x_al[i]}, sum:{s}")
#     dist_dict.append(s)
# dist_dict = np.array(dist_dict)
# with open('dist_dict_pretrain.npy', 'wb') as f:
#     np.save(f, dist_dict)
# print("dist_dict_pretrain saved!")

n_queries = 1700
mode = "lb2"
#queries_num = np.linspace(100, 3500, 35)
mse_dict = {}
all_chosen_samples_reg = []
all_chosen_samples_clf = []

mode_dict = {
    "random": random_sampling,
    "uncertainty": uncertainty_sampling,
    "margin": margin_sampling,
    "entropy": entropy_sampling,
    "lb": location_based_sampling,
    "lb2": location_based_sampling2
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

#_chosen_x, _chosen_y = np.array([]), np.array([])
#_new_x = copy.copy(train_x_al)
#_new_y = copy.copy(train_y_al)
# for i in range(n_queries + 1):
i = 0
pool_x_shape = pool_x_al.shape
best_mse = 999999999
best_iteration = -1
while i <= n_queries:
    # get_prediction_precision(regressor, x_test_ae_cnn, y_test_ae_cnn)
    if mode not in ["lb", "random"]:
        query_idx, query_instance = classifier.query(pool_x_al)
    else:
        query_idx, query_instance = regressor.query(pool_x_al)
    _x, _y_reg, _y_clf = pool_x_al[query_idx], pool_y_al[query_idx], pool_y_cluster[query_idx]
    pool_x_al = np.delete(pool_x_al, query_idx, axis=0)
    pool_y_al = np.delete(pool_y_al, query_idx, axis=0)
    pool_y_cluster = np.delete(pool_y_cluster, query_idx, axis=0)
    #print(f"_x:{_x}, _y:{_y}")
    all_chosen_samples_reg.append(_y_reg)
    all_chosen_samples_clf.append(_y_clf)

    xs.append(_x)
    ys_reg.append(_y_reg)
    ys_clf.append(_y_clf)
    xs_idx.append(query_idx[0])

    if i % 50 == 0 and i > 0:
        # batch mode AL teach
        regressor_copy = copy.copy(regressor)
        classifier_copy = copy.copy(classifier)
        #regressor.teach(_chosen_x[0], _chosen_y[0], only_new=False)

        # _model = copy.copy(model_ae)
        #_new_x = np.concatenate([_new_x, _x])
        #_new_y = np.concatenate([_new_y, _y])
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
        #model_ae.fit(xs, ys, validation_data=(x_test_ae_cnn, y_test_ae_cnn), verbose=1)

        # _model.fit(_new_x, _new_y, epochs=10, validation_data=(
        #     x_test_ae_cnn, y_test_ae_cnn), verbose=1)
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


        if i <= 300 and mse > 100000:
            print(f"Encountered large MSE: {mse}")
            i = i - 1
            regressor = regressor_copy
            classifier = classifier_copy
            xs = xs_backup
            ys_reg = ys_reg_backup
            ys_clf = ys_clf_backup
            xs_idx = xs_idx_backup
            continue
        elif i > 300 and mse > 40000:
            print(f"Encountered large MSE: {mse}")
            i = i - 1
            regressor = regressor_copy
            classifier = classifier_copy
            xs = xs_backup
            ys_reg = ys_reg_backup
            ys_clf = ys_clf_backup
            xs_idx = xs_idx_backup
            continue

        # y_pred = regressor.predict(x_test_ae_cnn)
        # mse = mean_squared_error(y_true=y_test_ae_cnn, y_pred=y_pred)
        mse_dict[i] = mse
        print(f"Evaluating model at {i}th query...mse:{mse}")
        xs, ys_reg, ys_clf, xs_idx = [], [], [], []
    i += 1

#print("training cnn model on ae...")
# lr = 5e-3
# epochs = 50
# decay_rate = lr / 50
# model_ae = cnn_model1(x_train_ae_cnn.shape[1:], opt=Adam(5e-3, decay=decay_rate))
# earlystopper = EarlyStopping(patience=50, verbose=1)
# #cp = ModelCheckpoint('v1_no_pca.h5', verbose=1, save_best_only=True)
# hist_ae = model_ae.fit(x_train_ae_cnn, y_train_ae_cnn, epochs=50, validation_data=(x_test_ae_cnn, y_test_ae_cnn),
#                  callbacks=[earlystopper], batch_size=4, verbose=1)

print("saving model...")
model_ae.save(f"models/model_ae_al3_{mode}_all.h5")
print("making predictions & saving results...")
preds = model_ae.predict(x_test_ae_cnn)
result = pd.DataFrame(preds, columns=["x", "y", "z"])
print(f"{len(result)} rows")
result.to_csv(f"preds/cnn_ae_al3_{mode}_all_preds.csv")
print("saving chosen samples")
df_chosen = pd.DataFrame(
    np.array(all_chosen_samples_reg).reshape(-1, 3), columns=["x", "y", "z"])
df_chosen.to_csv(f"chosen/cnn_ae_al3_{mode}_all_chosen.csv")
print(f"All done!! mode:{mode}")
print(f"mse_dict: {mse_dict}")
print(f"best iteration: {best_iteration}")
# print("saving model...")
# model0.save('model3.h5')
# print("All done!")
#val_loss = hist.history['val_loss']
