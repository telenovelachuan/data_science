from sklearn.datasets import fetch_openml
import numpy as np

mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
y = y.astype(np.uint8)
# print "y[0]: {}".format(y[0])

import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)

# plt.imshow(some_digit_image, cmap=mpl.cm.binary, interpolation="nearest")
# plt.axis("off")
# plt.show()

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
# binary classifier
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
# print "y_train_5: {}, {} items".format(y_train_5, len(y_train_5))

'''
Stochastic Gradient Descent, SGD
https://www.cnblogs.com/lliuye/p/9451903.html
'''
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
print "sgd_clf.predict([some_digit]): {}".format(sgd_clf.predict([some_digit]))

# self-implementing cross-validation
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
skfolds = StratifiedKFold(n_splits=3, random_state=42)
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    # print float(n_correct) / len(y_pred)

from sklearn.model_selection import cross_val_score
# print "cross_val_score: {}".format(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))

# confusion matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
# print "confusion matrix: {}".format(confusion_matrix(y_train_5, y_train_pred))
from sklearn.metrics import precision_score, recall_score
print "precision score: {}, recall score: {}".format(precision_score(y_train_5, y_train_pred), recall_score(y_train_5, y_train_pred))
from sklearn.metrics import f1_score
print "f1 score: {}".format(f1_score(y_train_5, y_train_pred))

y_scores = sgd_clf.decision_function([some_digit])
print "decision_function score: {}".format(y_scores)
threshold = 0
y_some_digit_pred = (y_scores > threshold)
print "y_some_digit_pred for threshold 0: {}".format(y_some_digit_pred)
threshold = 8000
y_some_digit_pred = (y_scores > threshold)
print "y_some_digit_pred for threshold 8000: {}".format(y_some_digit_pred)
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
print "y_scores: {}".format(y_scores)
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
# plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
# plt.show()

# get the threshold on certain precision
threshold_90_precision = thresholds[np.argmax(precisions >= 0.9)]  # ~7816
y_train_pred_90 = (y_scores >= threshold_90_precision)
print "precision_score: {}, recall_score: {}".format(precision_score(y_train_5, y_train_pred_90), recall_score(y_train_5, y_train_pred_90))

# ROC, Receiver Operating Characteristic, fpr on tpr
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
print "fpr:{}, tpr:{}, thresholds:{}".format(fpr, tpr, thresholds)
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')  # dashed diagonal
# plot_roc_curve(fpr, tpr)
# plt.show()
from sklearn.metrics import roc_auc_score
# print "roc_auc_score: {}".format(roc_auc_score(y_train_5, y_scores))

from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
y_scores_forest = y_probas_forest[:, 1]  # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)
# plt.plot(fpr, tpr, "b:", label="SGD")
# plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
# plt.legend(loc="lower right")
# plt.show()

# most binary classifier uses OvA, SVM uses OvO
sgd_clf.fit(X_train, y_train)
# print "sgd_clf for multiple classes: {}".format(sgd_clf.predict([some_digit]))
some_digit_scores = sgd_clf.decision_function([some_digit])
# print "some_digit_scores: {}".format(some_digit_scores)
# print "np.argmax(some_digit_scores): {}".format(np.argmax(some_digit_scores))
# print "sgd_clf.classes_: {}".format(sgd_clf.classes_)

forest_clf.fit(X_train, y_train)
# print "forest_clf for multiple classes:{}".format(forest_clf.predict([some_digit]))
# print "forest_clf.predict_proba([some_digit]): {}".format(forest_clf.predict_proba([some_digit]))

# print "cross_val_score for sgd_clf: {}".format(cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy"))
# scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
# print "cross_val_score for sgd_clf after scaling: {}".format(cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy"))

# Error Analysis
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
# print "confusion matrix: {}".format(conf_mx)
# plt.matshow(conf_mx, cmap=plt.cm.gray)
# plt.show()
row_sums = conf_mx.sum(axis=1, keepdims=True)
row_sums_float = [[float(x[0])] for x in row_sums]
# print "row_sums_float: {}".format(row_sums_float)

norm_conf_mx = conf_mx / row_sums_float
# print "norm_conf_mx: {}".format(norm_conf_mx)
np.fill_diagonal(norm_conf_mx, 0)
# plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
# plt.show()

# deep dive 3s and 5s
cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]
# plt.figure(figsize=(8, 8))
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap=plt.cm.binary, **options)
    plt.axis("off")
# plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
# plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
# plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
# plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
# plt.show()

# multilabel classification
# from sklearn.neighbors import KNeighborsClassifier
# y_train_large = (y_train >= 7)
# y_train_odd = (y_train % 2 == 1)
# y_multilabel = np.c_[y_train_large, y_train_odd]
# knn_clf = KNeighborsClassifier()
# knn_clf.fit(X_train, y_multilabel)
# print "knn_clf for multilabel: {}".format(knn_clf.predict([some_digit]))
# y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
# print "f1 score for knn: {}".format(f1_score(y_multilabel, y_train_knn_pred, average="macro"))

# multioutput classification
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test
print "noise added"

# display noised digit
# plt.imshow(X_test_mod[0].reshape(28, 28), cmap=mpl.cm.binary, interpolation="nearest")  # '7'
# plt.axis("off")

# noise removing
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[0]])
plot_digits(clean_digit)
plt.show()





