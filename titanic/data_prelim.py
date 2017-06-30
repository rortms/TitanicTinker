################################
# Titanic Survival Exploration

# Objective:
# Explore a subset of the RMS titanic passenger manifest to determine which features best predict whether
# someone survived or did not survive.

import numpy as np
import pandas as pd

# Make categorical into dummies
def cat2Dummies(X):
    outX = pd.DataFrame(index=X.index)  # output dataframe, initially empty
    #
    # Check each column
    for col, col_data in X.iteritems():
        #
        # For other categories convert to one or more dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)  # e.g. 'school' => 'school_GP', 'school_MS'
        #    
        outX = outX.join(col_data)  # collect column(s) in output dataframe
    return outX

data = pd.read_csv("titanic_data.csv")

##############
# Get features

# Create Cabin boolean feature (Did they have a cabin number or not)
has_cabin = pd.Series([1 if v == False else 0 for v in data['Cabin'].isnull()], )

# All relevant features (Cabin will be added back)
features = [f for f in data.columns if f not in
            ["Survived", 'Name', 'PassengerId', 'Cabin', 'Ticket' ] ]

X_all = pd.DataFrame(data[features], columns=features)

X_all['Cabin'] = has_cabin
y_all = pd.Series(data['Survived'])

good_indices = [i for i in X_all.index if not X_all['Age'].isnull()[i]]
X_all = X_all.iloc[good_indices]
y_all = y_all.iloc[good_indices]


## Sanity checks
# print reduce(lambda x,y: x and y, list(X_all['Cabin'] == has_cabin))
# NaNs = sum([1 for v in list(X_all['Cabin'].isnull()) if v == True])
# print 1.*NaNs/ X_all.shape[0] *100

###############################
# Make categorical into dummies
X_all = cat2Dummies(X_all)
# print X_all.describe()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all,
                                                    test_size= 0.25,
                                                    random_state = 123)

## Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

## Base line KNN Performance
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
print accuracy_score(clf.predict(X_test), y_test)

## Gridsearch Tuning
all_params = { 'KNeighborsClassifier': { 'n_neighbors' : range(3, 30, 3),
                                         'weights': ['uniform', 'distance'],
                                         'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                                         'leaf_size': [10,15,20,30,35] },
               
               'DecisionTreeClassifier': { 'criterion': ["entropy", "gini"],
                                           'max_features':["sqrt", "log2"],
                                           'random_state': [0],
                                           'max_depth': range(2,11),
                                           'min_samples_split':range(2,9),
                                           'min_samples_leaf':range(1,9) }
               }


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from time import time

clfs = [ KNeighborsClassifier(),
         DecisionTreeClassifier()]

# clfs = [ KNeighborsClassifier()]

scorer = make_scorer(accuracy_score)
grid_results = { 'KNeighborsClassifier' : {},
                 'DecisionTreeClassifier': {} }


for clf in clfs:
    name = clf.__class__.__name__
    
    grid_clf = GridSearchCV(clf, all_params[name], scorer, n_jobs=4, cv = 10)

    start = time()
    grid_clf = grid_clf.fit(X_train, y_train)
    grid_time = time() - start


    grid_results[name]['accuracy'] = accuracy_score(grid_clf.predict(X_test), y_test)
    grid_results[name]['params'] = grid_clf.best_params_
    grid_results[name]['grid time'] = "{} s".format(grid_time)

for r in grid_results:
    if grid_results[r] != {}:
        print r, ": "
        print "accuracy: ", grid_results[r]['accuracy']
        print "Grid time: ", grid_results[r]['grid time']
        print "Model parameters:", grid_results[r]['params']
        print
        print
