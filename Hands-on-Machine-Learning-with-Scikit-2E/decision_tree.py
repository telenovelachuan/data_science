from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data[:, 2:]  # petal length and width
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def get_prediction_precision(model, testing_data, testing_labels):
    y_pred = model.predict(testing_data)
    n_correct = sum(y_pred == testing_labels)
    print "For model:{}".format(model)
    print "n_correct:{}".format(n_correct)
    print "accuracy: {}".format(float(n_correct) / len(y_pred))

tree_clf = DecisionTreeClassifier(max_depth=2, criterion="entropy")
tree_clf.fit(X_train, y_train)
get_prediction_precision(tree_clf, X_test, y_test)

# export_graphviz(tree_clf, out_file="iris_tree.dot", feature_names=iris.feature_names[2:],
#                 class_names=iris.target_names, rounded=True, filled=True)
def visualize_decision_tree(tree_clf):
    export_graphviz(tree_clf, out_file="tree.dot", feature_names=iris.feature_names[2:],
                    class_names=iris.target_names, rounded=True, filled=True)
    from subprocess import call
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png'])
    plt.figure(figsize=(14, 7))
    plt.imshow(plt.imread('tree.png'))
    plt.axis('off')
    plt.show()
# visualize_decision_tree(tree_clf)

print "probabilities prediction for [[5, 1.5]]: {}".format(tree_clf.predict_proba([[5, 1.5]]))
print "class prediction for [[5, 1.5]]: {}".format(tree_clf.predict([[5, 1.5]]))

# decision tree regression
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(max_depth=3)
tree_reg.fit(X, y)
# visualize_decision_tree(tree_reg)

# Bagging: random forest, BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rnd_clf.fit(X, y)
get_prediction_precision(rnd_clf, X_test, y_test)


# feature importance
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rnd_clf.fit(iris["data"], iris["target"])
for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
    print name, score


# Boosting: AdaBoost
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1),
                             n_estimators=200, algorithm="SAMME.R", learning_rate=0.5)
ada_clf.fit(X, y)
get_prediction_precision(ada_clf, X_test, y_test)

'''
Boosting: GradientBoosting with early stopping
tries to fit the new predictor to the residual errors made by the previous predictor
'''

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120, learning_rate=1.0)
gbrt.fit(X_train, y_train)
errors = [mean_squared_error(y_test, y_pred) for y_pred in gbrt.staged_predict(X_test)]
print "errors:{}".format(errors)
bst_n_estimators = np.argmin(errors)
print "bst_n_estimators:{}".format(bst_n_estimators)
gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=bst_n_estimators)
gbrt_best.fit(X_train, y_train)
# plt.plot(range(len(errors)), errors)
# plt.show()

gbrt2 = GradientBoostingRegressor(max_depth=2, n_iter_no_change=5, learning_rate=1.0)
gbrt2.fit(X_train, y_train)
errors = [mean_squared_error(y_test, y_pred) for y_pred in gbrt2.staged_predict(X_test)]
print "errors:{}".format(errors)
# plt.plot(range(len(errors)), errors)
# plt.show()

# XGBoost
import xgboost
xgb_reg = xgboost.XGBClassifier(objective="multi:softmax")
xgb_reg.fit(X_train, y_train)
get_prediction_precision(xgb_reg, X_test, y_test)



