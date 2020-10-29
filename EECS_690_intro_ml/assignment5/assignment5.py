#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score


# In[81]:


import warnings
warnings.filterwarnings('ignore')


# In[11]:


df = pd.read_csv("imbalanced_iris.csv")
df_x = df.drop(columns=["class"])
df_y = df["class"]
df


# In[17]:


X_train, X_test, Y_train, Y_test = train_test_split(df_x, df_y, test_size=0.5, random_state=0)


# In[78]:


def evaluate_model(model_func, model_name, x_train=X_train, x_test=X_test,
                   y_train=Y_train, y_test=Y_test):
    #gnb = GaussianNB()
    model = model_func()
    y_pred1 = model.fit(x_train, y_train).predict(x_test)

    x_train, x_test = x_test, x_train
    y_train, y_test = y_test, y_train
    #gnb = GaussianNB()
    model = model_func()
    y_pred2 = model.fit(x_train, y_train).predict(x_test)

    y_pred = np.append(y_pred2, y_pred1)
    y_true = np.append(y_test, y_train)
    print(f"Accuracy: {round((y_pred == y_true).sum() / len(y_pred), 4)}")
    print(f"Confusion Matrix:\n {confusion_matrix(y_true, y_pred).T}")


# In[75]:


print("Part1.")
evaluate_model(lambda : MLPClassifier(), "Neural Network")


# In[76]:


precision1 = 40 / 40
precision2 = 26 / 26
precision3 = 50 / (50 + 4)
recall1 = 40 / 40
recall2 = 26 / (26 + 4)
recall3 = 50 / 50
class_balanced_acc = (min(precision1, recall1) + min(precision2, recall2) + min(precision3, recall3)) / 3
print(f"Class balanced accuracy: {class_balanced_acc}")


# In[77]:


specificity1 = (26 + 4 + 50) / (26 + 4 + 50 + 0)
specificity2 = (40 + 50) / (40 + 50 + 0)
specificity3 = (40 + 26) / (40 + 26 + 4)
balanced_acc = (np.mean([specificity1, recall1]) + np.mean([specificity1, recall2]) + np.mean([specificity1, recall3])) / 3
print(f"Balanced accuracy: {balanced_acc}")


# In[80]:


model = MLPClassifier()

y_pred1 = model.fit(X_train, Y_train).predict(X_test)

x_train, x_test = X_test, X_train
y_train, y_test = Y_test, Y_train
model = MLPClassifier()
y_pred2 = model.fit(x_train, y_train).predict(x_test)

y_pred = np.append(y_pred2, y_pred1)
y_true = np.append(y_test, y_train)
print(f"Scikit-learn balanced_accuracy_score: {balanced_accuracy_score(y_true, y_pred)}")


# In[62]:


print("Part2.")
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
x_resampled, y_resampled = ros.fit_resample(df_x, df_y)
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(x_resampled, y_resampled, test_size=0.5, random_state=0)

print("Random oversampling")
evaluate_model(lambda : MLPClassifier(), "Neural Network", x_train=X_train1, x_test=X_test1,
                   y_train=Y_train1, y_test=Y_test1)


# In[63]:


print("SMOTE oversampling")
from imblearn.over_sampling import SMOTE, ADASYN
x_resampled2, y_resampled2 = SMOTE().fit_resample(df_x, df_y)
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(x_resampled2, y_resampled2, test_size=0.5, random_state=0)
evaluate_model(lambda : MLPClassifier(), "Neural Network", x_train=X_train2, x_test=X_test2,
                   y_train=Y_train2, y_test=Y_test2)


# In[66]:


print("ADASYN oversampling")
from imblearn.over_sampling import SMOTE, ADASYN
try:
    x_resampled3, y_resampled3 = ADASYN().fit_resample(df_x, df_y)
    X_train3, X_test3, Y_train3, Y_test3 = train_test_split(x_resampled3, y_resampled3, test_size=0.5, random_state=0)
    evaluate_model(lambda : MLPClassifier(), "Neural Network", x_train=X_train3, x_test=X_test3,
                   y_train=Y_train3, y_test=Y_test3)
except Exception as ex:
    print(f"Error occurred while executing ADASYN oversampling:\n{str(ex)}")


# In[68]:


print("Part3.")
print("Random undersampling")
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
x_resampled4, y_resampled4 = rus.fit_resample(df_x, df_y)
X_train4, X_test4, Y_train4, Y_test4 = train_test_split(x_resampled4, y_resampled4, test_size=0.5, random_state=0)
evaluate_model(lambda : MLPClassifier(), "Neural Network", x_train=X_train4, x_test=X_test4,
                   y_train=Y_train4, y_test=Y_test4)


# In[71]:


print("Cluster undersampling")
from imblearn.under_sampling import ClusterCentroids
rus = ClusterCentroids(random_state=42)
x_resampled5, y_resampled5 = rus.fit_resample(df_x, df_y)
X_train5, X_test5, Y_train5, Y_test5 = train_test_split(x_resampled5, y_resampled5, test_size=0.5, random_state=0)
evaluate_model(lambda : MLPClassifier(), "Neural Network", x_train=X_train5, x_test=X_test5,
                   y_train=Y_train5, y_test=Y_test5)


# In[74]:


print("Tomek links undersampling")
from imblearn.under_sampling import TomekLinks
rus = TomekLinks()
x_resampled6, y_resampled6 = rus.fit_resample(df_x, df_y)
X_train6, X_test6, Y_train6, Y_test6 = train_test_split(x_resampled6, y_resampled6, test_size=0.5, random_state=0)
evaluate_model(lambda : MLPClassifier(), "Neural Network", x_train=X_train6, x_test=X_test6,
                   y_train=Y_train6, y_test=Y_test6)


# In[ ]:




