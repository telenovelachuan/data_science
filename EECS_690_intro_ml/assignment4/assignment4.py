#!/usr/bin/env python
# coding: utf-8

# In[96]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

from sklearn.decomposition import PCA
from numpy.linalg import eig
from numpy import array, mean, cov
import random
import math


# In[20]:


df, target = load_iris(return_X_y=True)
df = pd.DataFrame(df, columns=["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)"])
df["label"] = target
df_features = df.drop(columns=["label"])
df


# In[93]:


x, y = load_iris(return_X_y=True)



def evaluate_model(model_func, model_name, features, df_x=df_features, df_y=df["label"], verbose=1):
    #gnb = GaussianNB()
    model = model_func()
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.5, random_state=0)
    y_pred1 = model.fit(x_train, y_train).predict(x_test)

    x_train, x_test = x_test, x_train
    y_train, y_test = y_test, y_train
    #gnb = GaussianNB()
    model = model_func()
    y_pred2 = model.fit(x_train, y_train).predict(x_test)

    y_pred = np.round(np.append(y_pred2, y_pred1))
    y_true = np.append(y_test, y_train)
    if verbose == 1:
        print(f"Evaluating {model_name} model")
    acc = round((y_pred == y_true).sum() / len(y), 4)
    if verbose == 1:
        print(f"Accuracy: {acc}")
        print(f"Confusion Matrix:\n {confusion_matrix(y_true, y_pred, labels=[0, 1, 2])}")
        print(f"Features used: {', '.join(features)}")
    return acc


# In[94]:


# Part 1: original 4 features.
evaluate_model(lambda : DecisionTreeClassifier(), "Decision Tree", df_features.columns)


# In[49]:


# Part 2: PCA
features = ["p0", "p1", "p2", "p3"]
df_features = df.drop(columns=["label"])
pca = PCA(n_components=4)
df_pca = pd.DataFrame(pca.fit_transform(df_features), columns=features)

A = df_pca.values
M = mean(A.T, axis=1)
C = A - M
V = cov(C.T)
eigen_values, eigen_vectors = eig(V)
print(f"Eigen values: {eigen_values}, \nEigen vectors:\n{eigen_vectors}")
eigen_sum = sum(eigen_values)
povs = [round(v * 100 / eigen_sum, 2) for v in np.cumsum(eigen_values)]
povs_print = [f'{round(v * 100 / eigen_sum, 2)}%' for v in np.cumsum(eigen_values)]
print(f"PoVs: {', '.join(povs_print)}")

max_idx = len([f for idx, f in enumerate(features) if povs[idx] < 90])
feature_subset = features[:max_idx + 1]
print(f"Feature subset that makes PoV > 0.9: {', '.join(feature_subset)}")
evaluate_model(lambda : DecisionTreeClassifier(), "Decision Tree", feature_subset, df_x=df_pca[feature_subset])


# In[58]:


df_pca


# In[60]:


# Part 3: Simulated Annealing

df_sa = pd.concat([df, df_pca], axis=1)
df_sa_features = df_sa.drop(columns=["label"])
df_sa_features


# In[120]:


current_features = list(df_features.columns)
unused_features = list(df_pca.columns)
iterations = 100
previous_acc = -1
num_max_features = len(df_sa_features.columns)
restart_value = 10
restart_cnt = 0
current_best = -1
results = []

for i in range(iterations):
    _row = [i + 1]
    if len(current_features) <= 1:
        mode = "add"
    elif len(current_features) == num_max_features:
        mode = "delete"
    else:
        mode = "add" if random.random() > 0.5 else "delete"
    
    if mode == "delete":
        # randomly select 1 or 2 features to delete
        num = random.randrange(1, 3)
        while num >= len(current_features):
            # all deleted
            num = random.randrange(1, 3)
        ran_features = random.sample(current_features, num)
        _current_features = [f for f in current_features if f not in ran_features]
        _unused_features = unused_features + ran_features
        print(f"deleting: {ran_features}")
    else:
        # randomly select 1 or 2 features to add
        num = random.randrange(1,3)
        while num + len(current_features) > num_max_features:
            # exceeds max feature num
            num = random.randrange(1,3)
        ran_features = random.sample(unused_features, num)
        _current_features = current_features + ran_features
        _unused_features = [f for f in unused_features if f not in ran_features]
        print(f"adding: {ran_features}")
        
    print(f"using features: {_current_features}")
    _row.append(",".join(_current_features))
    acc = evaluate_model(lambda : DecisionTreeClassifier(), "Decision Tree", _current_features,
                         df_x=df_sa_features[_current_features], verbose=0)
    _row.append(acc)
    print(f"Accuracy: {acc}, previous acc: {previous_acc}")
    _status = ""
    if acc > previous_acc:
        # improved
        print("improved")
        current_features = _current_features
        unused_features = _unused_features
        previous_acc = acc
        restart_cnt = 0
        current_best = acc
        _row.extend(["-", "-"])
        _status = "Improved"
    else:
        expv = (-(i + 1) / 1) * ((previous_acc - acc) / previous_acc)
        pr_accept = math.exp(expv)
        print(f"Pr[accept] = {pr_accept}")
        _row.append(pr_accept)
        restart_cnt += 1
        rand_uni = random.random()
        _row.append(round(rand_uni, 3))
        if pr_accept > rand_uni:
            # accept
            print("accepted")
            current_features = _current_features
            unused_features = _unused_features
            previous_acc = acc
            _status = "Accepted"
        else:
            # reject
            print("discarded")
            previous_acc = acc
            _status = "Discarded"
    
    # restart mechanism
    if restart_cnt >= restart_value:
        print("10 iterations without improvement. Restarting!!!!!!!")
        previous_acc = current_best
        _status = "Restart"
    _row.append(_status)
    results.append(_row)

df_result = pd.DataFrame(results, columns=["Iteration", "Subset of features", "Acc", "Pr[accept]", "Random Uniform", "Status"])
df_result


# In[121]:


print(df_result)


# In[122]:


# Part 4: Genetic Algorithm
df_sa_features


# In[175]:


def mutate(s, all_features, mode=None):
    if mode is None:
        if len(s) <= 1:
            modes = ["add", "replace"]
        elif len(s) == len(all_features):
            modes = ["delete"]
        else:
            modes = ["add", "delete", "replace"]
        
        mode = random.choice(modes)
        
    if mode == "add":
        available_features = [f for f in all_features if f not in s]
        return set(list(s) + [random.choice(available_features)])
    elif mode == "delete":
        chosen_remove = random.choice(list(s))
        return set([f for f in s if f != chosen_remove])
    else:
        chosen_remove = random.choice(list(s))
        chosen_add = random.choice([f for f in all_features if f not in s])
        return set([f for f in s if f != chosen_remove] + [chosen_add])

mutate({1,2,3}, {4,5,6,7})


# In[172]:


list(df_sa_features.columns)


# In[215]:


iterations = 50
current_sets = [
    {"p0", "sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"},
    {"p0", "p1", "sepal width (cm)", "petal length (cm)", "petal width (cm)"},
    {"p0", "p1", "p2", "sepal width (cm)", "petal length (cm)"},
    {"p0", "p1", "p2", "p3", "sepal width (cm)"},
    {"p0", "p1", "p2", "p3", "sepal length (cm)"}
]
all_features = list(df_sa_features.columns)
# unions = []
deduped_unions = []
highest_accuracies = []
best_sets = []
    
for i in range(iterations):
    
    # generate crossover
    # 1. union & intersection
    unions, intersections, handled_hashes = [], [], []
    for idx1, s1 in enumerate(current_sets):
        for idx2, s2 in enumerate(current_sets):
            _hash = f"{min(idx1, idx2)}_{(max(idx1, idx2))}"
            if s1 == s2 or _hash in handled_hashes:
                continue
            unions.append(s1.union(s2))
            handled_hashes.append(_hash)
            intersections.append(s1.intersection(s2))

    crossovers = current_sets + unions + intersections
    #print(f"!!!crossovers:{crossovers}")

    # 2. mutate
    mutations = []
    for s in crossovers:
        mutations.append(mutate(s, all_features))

    all_candidates = mutations + crossovers
    #print(f"!!!all_candidates:{all_candidates}")

    # evaluate
    for _s in all_candidates:
        _features = list(_s)
        _acc = evaluate_model(lambda : DecisionTreeClassifier(), "Decision Tree", _features,
                              df_x=df_sa_features[_features], verbose=0)
        if len(highest_accuracies) < 5:
            highest_accuracies.append(_acc)
            best_sets.append(_s)
        else:
            current_worst = min(highest_accuracies)
            if current_worst < _acc:
                lowest_index = highest_accuracies.index(current_worst)
                highest_accuracies[lowest_index] = _acc
                best_sets[lowest_index] = _s
                #print(f"!!!best_sets{best_sets}")
                current_sets = best_sets
                #print(f"{_acc} among highest, replace worst with {_features}. {highest_accuracies}")
            else:
                pass
                #print(f"{_acc} lower than all 5 scores, discard.")
    
    index_best = highest_accuracies.index(max(highest_accuracies))
    best_s = best_sets[index_best]
    print(f"The {str(i)}th iteration, best features: {','.join(best_s)}, highest 5 accuracies: {','.join([str(a) for a in highest_accuracies])}")


# In[206]:


print(highest_accuracies)
best_sets


# In[145]:


s1 = set({2,1})
s2 = set({2,3})
s1.intersection(s2)


# In[207]:


",".join(s1)


# In[192]:


a=[1,5,5,7,8]
a.index(5)


# In[ ]:


from itertools import combinations

