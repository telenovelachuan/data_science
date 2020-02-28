import pandas as pd
import numpy as np
import math
import xgboost
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Flatten, BatchNormalization
from tensorflow.keras.models import Model, Sequential, model_from_json
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, PredefinedSplit
from hypopt import GridSearch
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("../data/raw/train.csv")
#df.dropna(inplace=True)
y_label = 'Survived'
#columns_to_remove = ['Name', 'PassengerId']
columns_to_remove = ['PassengerId', 'Ticket']
df.drop(columns=columns_to_remove, inplace=True)
df['Cabin'].fillna('mising', inplace=True)
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode(), inplace=True)
df.dropna(inplace=True)


#ctg_columns = ['Sex', 'Ticket', 'Cabin', 'Embarked']
ctg_columns = ['Embarked', 'Cabin']
num_columns = list(set(df.columns) - set(ctg_columns) - set(y_label))

nd_title = [i.split(",")[1].split(".")[0].strip() for i in df["Name"]]
df["Title"] = pd.Series(nd_title)
df["Title"] = df["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', math.nan], 'Rare')
df["Title"] = df["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
df["Title"] = df["Title"].astype(int)
df["Fsize"] = df["SibSp"] + df["Parch"] + 1
df['Single'] = df['Fsize'].map(lambda s: 1 if s == 1 else 0)
df['SmallF'] = df['Fsize'].map(lambda s: 1 if  s == 2  else 0)
df['MedF'] = df['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
df['LargeF'] = df['Fsize'].map(lambda s: 1 if s >= 5 else 0)
df.drop(columns=['Name'], inplace=True)

# Create new feature IsAlone from FamilySize
df['IsAlone'] = 0
df.loc[df['Fsize'] == 1, 'IsAlone'] = 1
# Create new feature CategoricalFare
df['CategoricalFare'] = pd.qcut(df['Fare'], 4)
df.loc[df['Fare'] <= 7.91, 'Fare'] = 0
df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare']   = 2
df.loc[df['Fare'] > 31, 'Fare'] = 3
df['Fare'] = df['Fare'].astype(int)
df['Sex'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
# Create new feature CategoricalAge
df['CategoricalAge'] = pd.cut(df['Age'], 5)
df.loc[df['Age'] <= 16, 'Age'] = 0
df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
df.loc[df['Age'] > 64, 'Age'] = 4
df.drop(columns=['CategoricalAge', 'CategoricalFare'], inplace=True)
df_clean = df.copy(deep=True)

y_label = 'Survived'
df_train = pd.get_dummies(df, columns=ctg_columns)

y_train = df_train[y_label]
num_columns = ['Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare' 'Title', 'Fsize', 'Single', 'SmallF', 'MedF', 'LargeF', 'IsAlone']
num_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])

column_to_nmlz = list(set(df_train.columns) - set([y_label]))
#df_train[column_to_nmlz] = num_pipeline.fit_transform(df_train[column_to_nmlz])
df_train[column_to_nmlz] = num_pipeline.fit_transform(df_train[column_to_nmlz])

train_set_normed, test_set_normed = train_test_split(df_train, test_size=0.2, random_state=42, stratify=y_train)
x_training_normed = train_set_normed.drop(columns=[y_label], axis=1)
x_test_normed = test_set_normed.drop(columns=[y_label], axis=1)
y_training_normed = train_set_normed[y_label]
y_test_normed = test_set_normed[y_label]
df_train_normed = df_train.drop(columns=[y_label])
y_all = df_train[y_label]


y_train = df_clean[y_label]
df_clean = pd.get_dummies(df_clean, columns=ctg_columns)

clean_train_set, clean_test_set = train_test_split(df_clean, test_size=0.2, random_state=42)
clean_x_training = clean_train_set.drop(columns=[y_label], axis=1)
clean_x_test = clean_test_set.drop(columns=[y_label], axis=1)
clean_y_training = clean_train_set[y_label]
clean_y_test = clean_test_set[y_label]


df_clean_all = df_clean.drop(columns=[y_label])
y_all = df_clean[y_label]

def get_prediction_precision(model, testing_data, testing_labels):
    y_pred = model.predict(testing_data)
    n_correct = sum(y_pred == testing_labels)
    print("n_correct:{}".format(n_correct))
    print("accuracy: {}".format(float(n_correct) / len(y_pred)))

df_test = pd.read_csv("../data/raw/test.csv")
df_test['Cabin'].fillna('mising', inplace=True)
df_test['Age'].fillna(df['Age'].mean(), inplace=True)
df_test['Fare'].fillna(df['Fare'].mean(), inplace=True)

test_final_columns = list(df_train.columns)
test_final_columns.remove(y_label)
df_test_dummy = pd.DataFrame(columns=test_final_columns)
for idx, row in df_test.iterrows():
    series_dict = {}
    for col in test_final_columns:
        if '_' in col:
            ctg_name, ctg_value = col.split('_')
            series_dict[col] = 1 if row[ctg_name] == ctg_value else 0
        else:
            series_dict[col] = row[col]
    df_test_dummy= df_test_dummy.append(series_dict, ignore_index=True)


df_test_dummy[column_to_nmlz] = num_pipeline.fit_transform(df_test_dummy[column_to_nmlz])


def dict_product(d):
    keys = d.keys()
    for element in product(*d.values()):
        yield dict(zip(keys, element))


def GridSearchWithVal(model_class, param_grid, sub_class=None, sub_param_grid=None):
    combinations = list(dict_product(param_grid))
    best_acc = 0
    best_comb, best_model = None, None
    best_sub_comb = None
    print("{} combinations in total".format(len(combinations)))
    for comb in combinations:
        print(comb)

        if sub_class is None:
            model = model_class(**comb)
            model.fit(x_training_normed, y_training_normed)
            # model.fit(clean_x_training, clean_y_training)
            y_pred = model.predict(x_test_normed)
            # y_pred = model.predict(clean_x_test)
            n_correct = sum(y_pred == y_test_normed)
            # n_correct = sum(y_pred == clean_y_test)
            acc = float(n_correct) / len(y_pred)
            if acc > best_acc:
                best_acc = acc
                best_comb = comb
                best_model = model
        else:
            sub_combinations = list(dict_product(sub_param_grid))
            for sub_comb in sub_combinations:
                estimator = sub_class(**sub_comb)
                model = model_class(estimator, **comb)
                # model.fit(X_training, y_training)
                model.fit(clean_x_training, clean_y_training)
                # y_pred = model.predict(X_test)
                y_pred = model.predict(clean_x_test)
                # n_correct = sum(y_pred == y_test)
                n_correct = sum(y_pred == clean_y_test)
                acc = float(n_correct) / len(y_pred)
                if acc > best_acc:
                    best_acc = acc
                    best_comb = comb
                    best_model = model
                    best_sub_comb = sub_comb

    print("best params:{}".format(best_comb))
    if best_sub_comb is not None:
        print("best_sub_comb:{}".format(best_sub_comb))
    print("best acc:{}".format(best_acc))
    return best_comb, best_sub_comb


rf_param_grid = {"n_estimators" : [150, 200, 250, 300, 250],
                  "criterion" : ["entropy"],
                  "max_depth": [4, 5, 6, 7, None],
                  "max_features": [None],
                  "min_samples_split": [2, 3],
                  "min_samples_leaf": [1, 3],
                  "bootstrap": [True, False]
                 }
rfc_best_comb, _ = GridSearchWithVal(RandomForestClassifier, rf_param_grid)

rfc_best_comb = {'n_estimators': 2000, 'criterion': 'mae', 'max_depth': None, 'max_features': 0.2, 'min_samples_split': 2, 'min_samples_leaf': 1, 'bootstrap': False}
rfc_best = RandomForestClassifier(**rfc_best_comb)
rfc_best.fit(x_training, y_training)
#rfc_best.fit(df_clean_all, y_all)
get_prediction_precision(rfc_best, x_test, y_test)
plot_AUC(rfc_best)
rfc_predicts = rfc_best.predict(clean_x_test)

y_pred = rf_best.predict(df_test_dummy)
df_output = pd.DataFrame(data={'PassengerId': df_test['PassengerId'], 'Survived': y_pred})
df_output.to_csv("../data/processed/submission.csv", index=False)


def create_keras_model(optimizer='adam', neuron=50, init='lecun_normal', act_alpha=0.01):
    model_sequences = [
        keras.layers.Dense(units=neuron, kernel_initializer=init,
                           # activation='relu',
                           input_shape=x_training_normed.shape[1:]
                           ),

        keras.layers.LeakyReLU(alpha=act_alpha)
    ]

    for i in range(5):
        model_sequences.append(keras.layers.Dense(units=neuron, kernel_initializer=init,
                                                  # activation='relu',
                                                  input_shape=x_training_normed.shape[1:]))
        model_sequences.append(keras.layers.LeakyReLU(alpha=act_alpha))

    # model_sequences.append(keras.layers.BatchNormalization())
    model_sequences.append(Dense(units=1, name='score_output',
                                 activation='sigmoid'
                                 ))
    # model_sequences.append(keras.layers.LeakyReLU(alpha=act_alpha))

    nn_model = keras.models.Sequential(model_sequences)
    nn_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return nn_model


optimizer = keras.optimizers.Adadelta(lr=1e-4)
nn_model = create_keras_model(optimizer=optimizer, neuron=100, init='lecun_uniform', act_alpha=0.05)
print("Keras model constructed")
early_stopping_cb = EarlyStopping(patience=20)
history = nn_model.fit(x_training_normed, y_training_normed, validation_data=(x_test_normed, y_test_normed), epochs=20000, callbacks=[early_stopping_cb], verbose=1)
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.show()

print(x_training_normed.shape)
nn_model.summary()
nn_model.get_weights()

def plot_AUC(model):
    preds = model.predict(x_test_normed)
    fpr, tpr, threshold = roc_curve(y_test_normed, preds)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

xgb_param_grid = {"objective" : ['binary:logistic', 'binary:hinge'],
                  "n_estimators": [50, 100, 150],
                  "base_score" : [0.03, 0.04, 0.05],
                  "max_depth": [3, 4, 5],
                  "gamma": [0.05, 0.1, 0.2],
                  "min_child_weight": range(1, 6, 2)
                 }
xgb_best_comb, _ = GridSearchWithVal(xgboost.XGBClassifier, xgb_param_grid)

xgb_best = xgboost.XGBClassifier(**xgb_best_comb)
print("begin training XGBoost...")
xgb_best.fit(df_train_normed, y_all, verbose=True)
print("training finished, get predictions...")
#dump(xgb_clf, '../models/xgb.joblib')
get_prediction_precision(xgb_best, x_test_normed, y_test_normed)
#print("score: {}".format(xgb_best.score(X_test, y_test)))
plot_AUC(xgb_best)
xgb_predicts = xgb_best.predict(x_test_normed)

xgb_clf.feature_importances_


def plot_feature_importance(model):
    indices = np.argsort(model.feature_importances_)[::-1][:20]
    g = sns.barplot(y=X_training.columns[indices][:40], x=model.feature_importances_[indices][:40], orient='h')
    g.set_xlabel("Relative importance", fontsize=12)
    g.set_ylabel("Features", fontsize=12)
    g.tick_params(labelsize=9)


plot_feature_importance(xgb_clf)

gb_param_grid = {"n_estimators" : [50, 100, 150, 200, 250],
                  "loss" : ["deviance"],
                  "learning_rate": [0.03, 0.04, 0.05, 0.06, 0.08, 0.1],
                  "max_depth": [4, 5, 6, 7],
                  "min_samples_leaf": [50, 100, 120],
                  "max_features": [0.3, 0.1, None]
                }
gbc_best_param, _ = GridSearchWithVal(GradientBoostingClassifier, gb_param_grid)

GB_best = GradientBoostingClassifier(**gbc_best_param)
GB_best.fit(df_train_normed, y_all)
get_prediction_precision(GB_best, x_test_normed, y_test_normed)
plot_AUC(GB_best)
gb_predicts = GB_best.predict(x_test_normed)
#plot_feature_importance(GB_clf)

gdtc = DecisionTreeClassifier()
adaDTC = AdaBoostClassifier(gdtc, random_state=17)
gb_param_grid = {"algorithm" : ["SAMME","SAMME.R"],
                 "n_estimators" :[200, 250, 280, 300, 350],
                 "learning_rate":  [0.03, 0.04, 0.05, 0.06, 0.07]
                }
sub_param_grid = {
    "criterion": ["gini", "entropy"],
    "splitter": ["best", "random"],
    "max_depth": [3, 4, 5, None],
    "min_samples_split": [1.0, 2,3,4]
}

gdt_best_params, dt_best_params = GridSearchWithVal(AdaBoostClassifier, gb_param_grid, DecisionTreeClassifier, sub_param_grid)

dtc = DecisionTreeClassifier(**dt_best_params)
adb_best = AdaBoostClassifier(base_estimator=dtc, **gdt_best_params)
adb_best.fit(df_clean_all, y_all)
get_prediction_precision(adb_best, x_test_normed, y_test_normed)
plot_AUC(adb_best)
adb_predicts = adb_best.predict(x_test_normed)
#plot_feature_importance(adb_best)

svm_param_grid = {'kernel': ['rbf'],
                  'probability': [True],
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]
                }
svm_best_param, _ = GridSearchWithVal(SVC, svm_param_grid)

svm_best = SVC(**svm_best_param)
svm_best.fit(df_clean_all, y_all)
get_prediction_precision(svm_best, clean_x_training, clean_y_training)
plot_AUC(svm_best)
svm_predicts = svm_best.predict(clean_x_test)

voting_clf = VotingClassifier(estimators=[
    ('rfc', rfc_best), ('xgb', xgbc_best), ('gbc', gbc_best), ('adb', adbc_best), ('svm', svm_best)], voting='soft')
voting_clf = voting_clf.fit(df_clean_all, y_all)
get_prediction_precision(voting_clf, clean_x_test, clean_y_test)
plot_AUC(voting_clf)
#plot_feature_importance(voting_clf)

ensemble_results = pd.concat([pd.Series(rfc_predicts), pd.Series(xgb_predicts), pd.Series(gb_predicts),
                              pd.Series(adb_predicts), pd.Series(svm_predicts)], axis=1)
g= sns.heatmap(ensemble_results.corr(),annot=True)

df_test = pd.read_csv("../data/raw/test.csv")

df_test['Cabin'].fillna('mising', inplace=True)
df_test['Age'].fillna(df['Age'].mean(), inplace=True)
df_test['Fare'].fillna(df['Fare'].mean(), inplace=True)
df_test['Embarked'].fillna(df['Embarked'].mode(), inplace=True)
df_test.isna().sum()

passenger_ids = df_test['PassengerId']
columns_to_remove = ['PassengerId', 'Ticket']
df_test.drop(columns=columns_to_remove, inplace=True)

nd_title_test = [i.split(",")[1].split(".")[0].strip() for i in df_test["Name"]]
df_test["Title"] = pd.Series(nd_title_test)
df_test["Title"] = df_test["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', math.nan], 'Rare')
df_test["Title"] = df_test["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
df_test["Title"] = df_test["Title"].astype(int)
df_test["Fsize"] = df_test["SibSp"] + df_test["Parch"] + 1
df_test['Single'] = df_test['Fsize'].map(lambda s: 1 if s == 1 else 0)
df_test['SmallF'] = df_test['Fsize'].map(lambda s: 1 if  s == 2  else 0)
df_test['MedF'] = df_test['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
df_test['LargeF'] = df_test['Fsize'].map(lambda s: 1 if s >= 5 else 0)
df_test.drop(columns=['Name'], inplace=True)

# Create new feature IsAlone from FamilySize
df_test['IsAlone'] = 0
df_test.loc[df_test['Fsize'] == 1, 'IsAlone'] = 1
# Create new feature CategoricalFare
df_test['CategoricalFare'] = pd.qcut(df_test['Fare'], 4)
df_test.loc[df_test['Fare'] <= 7.91, 'Fare'] = 0
df_test.loc[(df_test['Fare'] > 7.91) & (df_test['Fare'] <= 14.454), 'Fare'] = 1
df_test.loc[(df_test['Fare'] > 14.454) & (df_test['Fare'] <= 31), 'Fare']   = 2
df_test.loc[df_test['Fare'] > 31, 'Fare'] = 3
df_test['Fare'] = df_test['Fare'].astype(int)
df_test['Sex'] = df_test['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
# Create new feature CategoricalAge
df_test['CategoricalAge'] = pd.cut(df_test['Age'], 5)
df_test.loc[df_test['Age'] <= 16, 'Age'] = 0
df_test.loc[(df_test['Age'] > 16) & (df_test['Age'] <= 32), 'Age'] = 1
df_test.loc[(df_test['Age'] > 32) & (df_test['Age'] <= 48), 'Age'] = 2
df_test.loc[(df_test['Age'] > 48) & (df_test['Age'] <= 64), 'Age'] = 3
df_test.loc[df_test['Age'] > 64, 'Age'] = 4
df_test.drop(columns=['CategoricalAge', 'CategoricalFare'], inplace=True)


test_final_columns = list(df_clean.columns)
test_final_columns.remove(y_label)
df_test_dummy = pd.DataFrame(columns=test_final_columns)
for idx, row in df_test.iterrows():
    series_dict = {}
    for col in test_final_columns:
        if '_' in col:
            ctg_name, ctg_value = col.split('_')
            series_dict[col] = 1 if row[ctg_name] == ctg_value else 0
        else:
            series_dict[col] = row[col]
    df_test_dummy= df_test_dummy.append(series_dict, ignore_index=True)


#df_test_dummy[column_to_nmlz] = num_pipeline.fit_transform(df_test_dummy[column_to_nmlz])


y_pred_9 = voting_clf.predict(df_test_dummy)
df_output_9 = pd.DataFrame(data={'PassengerId': passenger_ids, 'Survived': y_pred_9})
df_output_9
df_output_9.to_csv("../data/processed/submission_9.csv", index=False)





