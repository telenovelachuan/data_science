import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
# X = iris["data"][:, :2]  # petal length, petal width
# y = (iris["target"] == 2).astype(np.float64)  # Iris-Virginica
y = iris.target
print "X:{}".format(X)

linear_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=0.1, loss="hinge"))
])
linear_svm_clf.fit(X, y)
print "SVM prediction for [[5.5, 1.7]]: {}".format(linear_svm_clf.predict([[5.5, 1.7]]))

# nonlinear SVM
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures
polynomial_svm_clf = Pipeline([
    ("poly_features", PolynomialFeatures(degree=3)),
    ("scaler", StandardScaler()),
    ("svm_clf", LinearSVC(C=10, loss="hinge"))
])
polynomial_svm_clf.fit(X, y)

# using kernel
from sklearn.svm import SVC
poly_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
])
poly_kernel_svm_clf.fit(X, y)
X1_y0 = []
X2_y0 = []
X1_y1 = []
X2_y1 = []
X1_y2 = []
X2_y2 = []
for (x_sample, y_sample) in zip(X, y):
    if y_sample == 0:
        X1_y0.append(x_sample[0])
        X2_y0.append(x_sample[1])
    elif y_sample == 1:
        X1_y1.append(x_sample[0])
        X2_y1.append(x_sample[1])
    else:
        X1_y2.append(x_sample[0])
        X2_y2.append(x_sample[1])
plt.scatter(X1_y0, X2_y0, color="red", marker='^')
plt.scatter(X1_y1, X2_y1, color="blue", marker='o')
plt.scatter(X1_y2, X2_y2, color="green", marker='s')

def plot_svm(x, attr1, attr2, clf):
    h = .02  # step size in the mesh
    x_min, x_max = x[:, attr1].min() - 1, x[:, attr1].max() + 1
    y_min, y_max = X[:, attr2].min() - 1, x[:, attr2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.show()
# plot_svm(X, 0, 1, poly_kernel_svm_clf)

# Gaussian RBF kernel
rbf_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="rbf", gamma=0.1, C=5))
])
rbf_kernel_svm_clf.fit(X, y)
# plot_svm(X, 0, 1, rbf_kernel_svm_clf)
# plot_svm(X, 0, 1, linear_svm_clf)

# SVM regression
from sklearn.svm import LinearSVR
svm_reg = LinearSVR(epsilon=1.5)
svm_reg.fit(X, y)
from sklearn.svm import SVR
svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
svm_poly_reg.fit(X, y)




