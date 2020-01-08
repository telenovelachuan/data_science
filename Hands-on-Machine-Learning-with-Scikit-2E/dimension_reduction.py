import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt


mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
y = y.astype(np.uint8)

pca = PCA(n_components=2)
X2D = pca.fit_transform(X)
print "pca.components_:{}".format(pca.components_)
print "pca.explained_variance_ratio_:{}".format(pca.explained_variance_ratio_)

# setting the number of dimensions
pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1

pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)
# plt.plot(cumsum)
# plt.show()

# randomized PCA
rnd_pca = PCA(n_components=154, svd_solver="randomized")
X_reduced = rnd_pca.fit_transform(X)
# print "pca.components_:{}".format(rnd_pca.components_)
# print "pca.explained_variance_ratio_:{}".format(rnd_pca.explained_variance_ratio_)

# incremental PCA, for large datasets
from sklearn.decomposition import IncrementalPCA
n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X, n_batches):
    inc_pca.partial_fit(X)
X_reduced = inc_pca.transform(X)

# kernel PCA and grid search
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import KernelPCA
clf = Pipeline([
    ("kpca", KernelPCA(n_components=2)),
    ("log_reg", LogisticRegression())
])
param_grid = [{
    "kpca__gamma": np.linspace(0.03, 0.05, 10),
    "kpca__kernel": ["rbf", "sigmoid"]
}]
grid_search = GridSearchCV(clf, param_grid, cv=3, verbose=1)
grid_search.fit(X, y)
print "best params for kPCA:{}".format(grid_search.best_params_)



