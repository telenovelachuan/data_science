import numpy as np
import matplotlib.pyplot as plt


X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
print "y:{}".format(X)
# plt.scatter(X, y,  label="test")
# plt.show()

# Stochastic Gradient Descent
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())
print "SGD intercept: {}, SGD coef: {}".format(sgd_reg.intercept_, sgd_reg.coef_)

# Polynomial regression
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
# plt.scatter(X, y, label="test")
# plt.show()
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
print "X_poly[0]: {}".format(X_poly[0])
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
print "polynomial intercept: {}, polynomial coef: {}".format(lin_reg.intercept_, lin_reg.coef_)
X_poly_sorted = np.sort(X_poly, axis=1)
print "X_poly_sorted: {}".format(X_poly_sorted)
y_pred = lin_reg.predict(X_poly)
print "y_pred:{}".format(y_pred)
# plt.scatter(X, y_pred, color="r")
# plt.plot(X_poly_sorted, y_pred, label="line")
# plt.show()

# learning curve
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.plot(np.sqrt(train_errors), 'r-+', linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), 'b-', linewidth=3, label="val")
    plt.show()

lin_reg = LinearRegression()
# plot_learning_curves(lin_reg, X, y)

# learning curve of a 10th degree polynomial model
from sklearn.pipeline import Pipeline
polynomial_regression = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression())
])
# plot_learning_curves(polynomial_regression, X, y)

# Regularization
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X, y)
print "prediction by ridge regularization: {}".format(ridge_reg.predict([[1.5]]))
sgd_reg = SGDRegressor(penalty="l2")
sgd_reg.fit(X, y.ravel())
print "prediction by SGD with l2 regularization: {}".format(sgd_reg.predict([[1.5]]))

from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)   # or SGDRegressor(penalty="l1")
lasso_reg.fit(X, y)
print "prediction by lasso regularization: {}".format(lasso_reg.predict([[1.5]]))

from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
print "prediction by elastic net regularization: {}".format(elastic_net.predict([[1.5]]))

# early stopping
# prepare the data
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
poly_scaler = Pipeline([
    ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
    ("std_scaler", StandardScaler())
])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
X_train_poly_scaled = poly_scaler.fit_transform(X_train)
X_val_poly_scaled = poly_scaler.transform(X_val)

sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True,
                       penalty=None, learning_rate="constant", eta0=0.0005)
minimum_val_error = float("inf")
best_epoch = None
best_model = None
for epoch in range(1000):
    sgd_reg.fit(X_train_poly_scaled, y_train) # continues where it left off
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)
    val_error = mean_squared_error(y_val, y_val_predict)
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = clone(sgd_reg)

# logistic regression
from sklearn import datasets
iris = datasets.load_iris()
print "Iris dataset keys: {}".format(list(iris.keys()))
X = iris["data"][:, 3:]  # petal width
y = (iris["target"] == 2).astype(np.int)  # 1 if Iris-Virginica, else 0
print "Iris-y: {}".format(y)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X, y)
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
# plt.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica")
# plt.plot(X_new, y_proba[:, 0], "b--", label="not Iris-Virginica")
# plt.show()
print "LR prediction for [[1.7], [1.5]]: {}".format(log_reg.predict([[1.7], [1.5]]))

# softmax regression
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = iris["target"]
softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
softmax_reg.fit(X, y)
print "softmax prediction for [[5, 2]]: {}".format(softmax_reg.predict([[5, 2]]))
print "softmax proba prediction for [[5, 2]]: {}".format(softmax_reg.predict_proba([[5, 2]]))








