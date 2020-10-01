import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score
from dbn import SupervisedDBNClassification

df, target = load_iris(return_X_y=True)
df = pd.DataFrame(df, columns=["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)"])
df["label"] = target

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X = df.drop(columns=["label"])
Y = df["label"]
X = ss.fit_transform(X)

#x, y = load_iris(return_X_y=True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


def evaluate_model(model_func, model_name, x_train=X_train, x_test=X_test, y_train=Y_train, y_test=Y_test):
    #gnb = GaussianNB()
    model = model_func()
    y_pred1 = model.fit(x_train, y_train).predict(x_test)

    x_train, x_test = x_test, x_train
    y_train, y_test = y_test, y_train
    #gnb = GaussianNB()
    model = model_func()
    y_pred2 = model.fit(x_train, y_train).predict(x_test)

    y_pred = np.round(np.append(y_pred2, y_pred1))
    y_true = np.append(y_test, y_train)
    print(f"Evaluating {model_name} model")
    print(f"Accuracy: {round((y_pred == y_true).sum() / len(y), 4)}")
    print(f"Confusion Matrix:\n {confusion_matrix(y_true, y_pred, labels=[0, 1, 2])}")


dbn_clf = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
learning_rate_rbm=0.05,
learning_rate=0.1,
n_epochs_rbm=10,
n_iter_backprop=100,
batch_size=32,
activation_function='relu',
dropout_p=0.2)

dbn_clf.fit(X_train, Y_train)
Y_pred = dbn_clf.predict(X_test)
print('Done.\nAccuracy: %f' % accuracy_score(Y_test, Y_pred))
