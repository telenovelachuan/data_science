#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


# Loading Iris dataset

# In[5]:


df, target = load_iris(return_X_y=True)
df = pd.DataFrame(df, columns=["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)	"])
df["label"] = target
df


# Train-test split and mode evaluation function

# In[92]:


x, y = load_iris(return_X_y=True)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.5, random_state=0)


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


# Naïve Baysian

# In[94]:


evaluate_model(lambda : GaussianNB(), "Naïve Baysian")


# Linear Regression

# In[77]:


evaluate_model(lambda : LinearRegression(), "Linear Regression")


# Polynomial of degree 2 regression

# In[83]:


poly = PolynomialFeatures(2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)
evaluate_model(lambda : LinearRegression(), "Polynomial of degree 2 regression",
               x_train=X_train_poly, x_test=X_test_poly)


# Polynomial of degree 3 regression

# In[86]:


poly = PolynomialFeatures(3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)
evaluate_model(lambda : LinearRegression(), "Polynomial of degree 3 regression",
               x_train=X_train_poly, x_test=X_test_poly)


# KNN Classifier

# In[87]:


evaluate_model(lambda : KNeighborsClassifier(n_neighbors=3), "KNN Classifier")


# Linear Discriminant Analysis

# In[89]:


evaluate_model(lambda : LinearDiscriminantAnalysis(), "Linear Discriminant Analysis")


# Quadratic Discriminant Analysis

# In[91]:


evaluate_model(lambda : QuadraticDiscriminantAnalysis(), "Quadratic Discriminant Analysis")


# In[ ]:




