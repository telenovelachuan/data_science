# <--- Decision tree --->
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.preprocessing import LabelEncoder
import pandas

dtc = DTC(criterion='entropy')
df = pandas.read_csv('training_set.csv')

# encode labels with value between 0 and c_classes - 1
df = df.apply(LabelEncoder().fit_transform)
X = df.iloc[:, 2:-2].as_matrix()
Y = df.iloc[:, -1].as_matrix()


#dtc.fit(X, Y)
# from sklearn.tree import export_graphviz
# from sklearn.externals.six import StringIO
# with open("decision_tree.dot", 'w') as f:
#     f = export_graphviz(dtc, out_file=f)


# <--- Random Forest --->
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV

clf = RandomForestClassifier(n_estimators=8)
#feature_train, feature_test, target_train, target_test = train_test_split(X, Y, test_size=0.02)
feature_train, target_train = X, Y
feature_test, target_test = df.iloc[600:700, 2:-2].as_matrix(), df.iloc[600:700, -1].as_matrix()
s = clf.fit(feature_train, target_train)
print s
r = clf.score(feature_test, target_test)
print 'all feature:{}'.format(r)

#feature_train, feature_test, target_train, target_test = train_test_split(X[:, :-50], Y, test_size=0.02)
# feature_test, target_test = df.iloc[100:200, :-50].as_matrix(), df.iloc[100:200, -1].as_matrix()
# s = clf.fit(feature_train, target_train)
# r = clf.score(feature_test, target_test)
# print 'partial feature:{}'.format(r)

parameter_space = {
    #"max_features": [2, 10, 'auto'],
    "n_estimators": [80],
    "criterion": ["gini", "entropy"],
    "min_samples_leaf": [2, 4, 6]
}
# grid = GridSearchCV(clf, parameter_space)
# grid.fit(X, Y)
# print 'GridSearchCV:{}'.format(grid.best_score_)


