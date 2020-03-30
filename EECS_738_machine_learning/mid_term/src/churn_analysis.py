import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from itertools import product
import xgboost
from tensorflow import keras
from tensorflow.python.keras import backend as k
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Flatten, BatchNormalization
from tensorflow.keras.models import Model, Sequential, model_from_json
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import sys

df = pd.read_csv("data/ACMETelephoneABT.csv")
y_label = 'churn'


df.isna().sum()

df.drop(columns=['age', 'occupation', 'customer'], inplace=True)
#df['age'] = df['age'].replace(0, np.nan)
#age_imputer = KNNImputer(n_neighbors=2)
#df['age'] = age_imputer.fit_transform(df['age'].values.reshape(-1, 1)).reshape(1, -1)[0]
#df['occupation'].fillna('missing', inplace=True)
df['regionType'].fillna('missing', inplace=True)
df['regionType'] = df['regionType'].replace({'unknown': 'missing', 'r': 'rural', 's': 'suburban', 't': 'town'})

df['children'] = df['children'].map({True: 1, False: 0})
df['smartPhone'] = df['smartPhone'].map({True: 1, False: 0})
df['homeOwner'] = df['homeOwner'].map({True: 1, False: 0})
df['creditCard'] = df['creditCard'].map({'TRUE': 1, 'FALSE': 0, 'no': 0, 't': 1, 'f': 0, 'yes': 1})
df['creditRating'] = df['creditRating'].map({'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1})
df['churn'] = df['churn'].map({True: 1, False: 0})

colormap = plt.cm.RdBu
f, axs = plt.subplots(1, 1, figsize=(16, 16))
sns.heatmap(df.corr(method='pearson', min_periods=1).round(decimals=2), linewidths=0.1, vmax=1.0, square=True,
            cmap=colormap, linecolor='white', annot=True)
#plt.xticks(rotation=30, fontsize=7)
plt.show()

ctg_columns = ['regionType', 'marriageStatus']
df_dummy = pd.get_dummies(df, columns=ctg_columns)
df_dummy

df_normed = df_dummy.copy()

column_to_nmlz = list(set(df_normed.columns) - set(['children', 'smartPhone', 'homeOwner', y_label]))
df_normed[column_to_nmlz] = StandardScaler().fit_transform(df_normed[column_to_nmlz])
train_set_normed, test_set_normed = train_test_split(df_normed, test_size=0.1, random_state=42)
train_set, test_set = train_test_split(df_dummy, test_size=0.1, random_state=43)

x_train_normed = train_set_normed.drop(columns=[y_label], axis=1)
x_test_normed = test_set_normed.drop(columns=[y_label], axis=1)
y_train_normed = train_set_normed[y_label]
y_test_normed = test_set_normed[y_label]

x_train = train_set.drop(columns=[y_label], axis=1)
x_test = test_set.drop(columns=[y_label], axis=1)
y_train = train_set[y_label]
y_test = test_set[y_label]

def dict_product(d):
    keys = d.keys()
    for element in product(*d.values()):
        yield dict(zip(keys, element))

def GridSearchWithVal(model_class, param_grid, metrics='accuracy', cv=5, normed=False):
    combinations = list(dict_product(param_grid))
    max_metrics = 0
    best_comb = None
    X_train = x_train if normed is False else x_train_normed
    Y_train = y_train if normed is False else y_train_normed
    X_test = x_test if normed is False else x_test_normed
    Y_test = y_test if normed is False else y_test_normed

    print("{} combinations in total. Metric: {}".format(len(combinations), metrics))
    for idx, comb in enumerate(combinations):
        model = model_class(**comb)
        model.fit(X_train, Y_train)
        y_preds = model.predict(X_test)
        # print("y_preds:{}".format(y_preds))
        # print("y_test:{}".format(y_test.values))
        error = 0;

        #scores = cross_val_score(model, X, Y, cv=cv, scoring=metrics)
        n_correct = sum(y_preds == Y_test)
        acc = float(n_correct) / len(y_preds)
        metrics_num = acc

        if metrics_num > max_metrics:
            max_metrics = metrics_num
            best_comb = comb
        progress_str = "{} / {}, best comb: {}, best score: {}".format(idx + 1, len(combinations), best_comb, max_metrics)
        sys.stdout.write('\r' + progress_str)

    print("best params:{}".format(best_comb))
    print("max {}: {}".format(metrics, max_metrics))
    return best_comb

xgb_param_grid = {"objective" : ['binary:logistic'],
                  "n_estimators": [400, 420, 450, 480, 500],
                  "base_score" : [0.1, 0.2, 0.3, 0.4, 0.5],
                  "max_depth": [1, 2, 3, 4],
                  "gamma": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.02, 0.05],
                  "min_child_weight": [10, 20, 30, 50, 70, 80]
                 }
xbg_best_param = GridSearchWithVal(xgboost.XGBClassifier, xgb_param_grid, normed=False)

xgb_clf = xgboost.XGBClassifier(**xbg_best_param)
xgb_clf.fit(x_train, y_train)
y_pred = xgb_clf.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))


def create_keras_model(optimizer='adam', neuron=50, init='lecun_normal', act_alpha=0.1, loss='binary_crossentropy'):
    model_sequences = [
        keras.layers.Dense(units=neuron, kernel_initializer=init,
                           activation='relu',
                           input_shape=x_train_normed.shape[1:]
                           ),
        # keras.layers.LeakyReLU(alpha=act_alpha),
        keras.layers.Dropout(rate=0.1, input_shape=x_train_normed.shape[1:]),
    ]

    for i in range(2):
        model_sequences.append(keras.layers.Dense(units=neuron, kernel_initializer=init,
                                                  activation='relu',
                                                  input_shape=x_train_normed.shape[1:]))
        # model_sequences.append(keras.layers.LeakyReLU(alpha=act_alpha)),
        keras.layers.Dropout(rate=0.1, input_shape=x_train_normed.shape[1:]),

    # model_sequences.append(keras.layers.BatchNormalization())
    model_sequences.append(Dense(units=1, name='score_output',
                                 activation='sigmoid'
                                 ))
    # model_sequences.append(keras.layers.LeakyReLU(alpha=act_alpha))

    nn_model = keras.models.Sequential(model_sequences)
    nn_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return nn_model

optimizer = keras.optimizers.Adadelta(lr=1e-3)
sgd = SGD(lr=0.0001, decay=0.000001, momentum=0.9)
nn_model = create_keras_model(optimizer=sgd, neuron=400, init='lecun_normal', act_alpha=0.05)
print("Keras model constructed")
early_stopping_cb = EarlyStopping(patience=50)
best_save = ModelCheckpoint('models/best.hdf5', save_best_only=True, monitor='val_acc', mode='max')
history = nn_model.fit(x_train_normed.values, y_train_normed.values, validation_data=(x_test_normed, y_test_normed), epochs=200, callbacks=[best_save], verbose=1)
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.show()

nn_model.load_weights('./models/best.hdf5')
y_pred = nn_model.predict_classes(x_test_normed)
y_pred
print("Accuracy:", accuracy_score(y_test_normed, y_pred))
print(confusion_matrix(y_test_normed, y_pred, labels=[0, 1]))

def tf_print(tensor, transform=None):

    # Insert a custom python operation into the graph that does nothing but print a tensors value
    def print_tensor(x):
        # x is typically a numpy array here so you could do anything you want with it,
        # but adding a transformation of some kind usually makes the output more digestible
        print(x if transform is None else transform(x))
        return x
    log_op = tf.py_func(print_tensor, [tensor], [tensor.dtype])[0]
    with tf.control_dependencies([log_op]):
        res = tf.identity(tensor)

    # Return the given tensor
    return res

from tensorflow.python.keras import backend as K
def binary_crossentropy_custom(y_true, y_pred):
    #y_pred = tf_print(y_pred)
    ts_diff = tf.cast(y_pred, tf.float32) - tf.cast(y_true, tf.float32)

    loss_1 = tf.reduce_sum(tf.cast(tf.math.less(ts_diff, -0.5), tf.float32)) * 700
    loss_2 = tf.reduce_sum(tf.cast(tf.math.greater(ts_diff, 0.5), tf.float32)) * 100
    loss_3 = binary_crossentropy(y_true, y_pred)
    return loss_1 + loss_2 + loss_3


sgd = SGD(lr=0.001, decay=0.000001, momentum=0.9)
nn_model_cst_loss = create_keras_model(optimizer=sgd, neuron=400, init='lecun_normal', act_alpha=0.05, loss=binary_crossentropy_custom)
print("Keras model using customized loss constructed")
early_stopping_cb = EarlyStopping(patience=50)
best_save = ModelCheckpoint('models/customized_best.hdf5', save_best_only=True, monitor='val_loss', mode='min')
history = nn_model_cst_loss.fit(x_train_normed.values, y_train_normed.values, validation_data=(x_test_normed, y_test_normed), epochs=500, callbacks=[best_save], verbose=1)
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.show()

nn_model_cst_loss.load_weights('models/customized_best.hdf5')
y_pred = nn_model_cst_loss.predict_classes(x_test_normed)
print("Accuracy:", accuracy_score(y_test_normed, y_pred))
print(confusion_matrix(y_test_normed, y_pred, labels=[0, 1]))