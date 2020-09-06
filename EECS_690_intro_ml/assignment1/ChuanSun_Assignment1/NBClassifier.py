import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
label = "class"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', label]
df = pd.read_csv(url, names=names)

# Train-test split
train_set, test_set = train_test_split(df, test_size=0.5, random_state=42)
x_train = train_set.drop(columns=[label])
y_train = train_set[label]
x_test = test_set.drop(columns=[label])
y_test = test_set[label]

# Train NBClassifier
nb_clf = GaussianNB()
nb_clf.fit(x_train, y_train)
y_pred = nb_clf.predict(x_test)

# Overall accuracy
print(f"Overall accuracy: {round(sum(y_pred==y_test) * 100 / len(y_test), 2)}%")

# Confusion matrix
labels = ["Iris-versicolor", "Iris-setosa", "Iris-virginica"]
df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred, labels=labels), columns=labels, index=labels)
print(f"\nConfusion matrix:\n{df_cm}")

# Precision, recall and F1 score
print(f"\nP, R, F1 scores:")
prf = precision_recall_fscore_support(y_test, y_pred, labels=labels)
for idx, label in enumerate(labels):
    print(f"For class {label}, precision: {prf[0][idx]}, recall: {prf[1][idx]}, F1: {prf[2][idx]}")



