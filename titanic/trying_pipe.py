################################
# Titanic Survival Exploration

# Objective:
# Explore a subset of the RMS titanic passenger manifest to determine which features best predict whether
# someone survived or did not survive.

###############
# Imports
###
import numpy as np
import pandas as pd
from time import time

## Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
## Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

## Tuning
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

## Pipe
from sklearn.pipeline import Pipeline

################

##################
# Helper Functions

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


## Read Data
data = pd.read_csv("titanic_data.csv")


#########################################
# Get features and Dummy if necessary

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

# ENd Pre-process
##############################################################################


#pca_n_tree = Pipeline([('pca', PCA()),
##########################################
# Preliminary Pipe DecisionTree/RandomForest


# print
# best_feats, max_acc = None, 0 
# for i in range(1,12):
#     select = SelectKBest(k=i)
#     # clf = DecisionTreeClassifier()
#     clf = RandomForestClassifier()
#     ​
#     steps = [('feature_selection', select),
#             ('random_forest', clf)]
#     ​
#     pipeline = Pipeline(steps)

#     X_train, X_test, y_train, y_test = train_test_split(X_all, y_all,
#                                                         test_size= 0.25,
#                                                         random_state = 123)

#     pipeline.fit(X_train, y_train)
#     y_prediction = pipeline.predict(X_test)
#     acc_score = accuracy_score(y_prediction, y_test)
#     if acc_score > max_acc:
#         best_feats = select.get_support(indices=True)

#     print "{} best features accuracy: {}".format(i, acc_score), [X_train.columns[k] for k in select.get_support(indices=True)]

    
# ## Decision Tree    
# ## 3 best features accuracy: 0.793296089385
# ['Pclass', 'Sex_female', 'Sex_male']
# ## 9 best features accuracy: 0.798882681564
# ['Pclass', 'Sex_female', 'Sex_male', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Cabin']


# ## Forest 
# ## 10 best features accuracy: 0.826815642458
# ['Pclass', 'Sex_female', 'Sex_male', 'Age', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Cabin']


####################
# Gridsearch Tuning

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all,
                                                    test_size= 0.25,
                                                    random_state = 123)

clf_params = { 'DecisionTreeClassifier' :   { 'criterion': ["entropy", "gini"],},
                                              # 'max_features':["sqrt", "log2"],
                                              # 'random_state': [0],
                                              # 'max_depth': range(2,11),
                                              # 'min_samples_split':range(2,9),
                                              # 'min_samples_leaf':range(1,9) },
               
               'RandomForestClassifier' :  { 'n_estimators' : range(2,7),}
                                             # 'criterion': ["entropy", "gini"],
                                             # 'max_features':["sqrt", "log2"],
                                             # 'random_state': [0],
                                             # 'max_depth': range(2,11),
                                             # 'min_samples_split':range(2,9),
                                             # 'min_samples_leaf':range(1,9) }
               }



clf_name = 'RandomForestClassifier'
clf = RandomForestClassifier()

steps = [('selector', SelectKBest()),
         ('classr', clf)]

# params_for_grid = { 'selector__k' : range(2,12) }
# params_for_grid.update(
#     {'classr__'+key : clf_params[clf_name][key] for key in clf_params[clf_name].keys()} )

# print( params_for_grid)
params_for_grid = {'selector__k': range(2, 12),
                   'classr__n_estimators': range(2, 7)}

pipeline = Pipeline(steps)

for p in pipeline.get_params():
    print( p)
    
scorer = make_scorer(accuracy_score)


grid_clf = GridSearchCV(clf,
                        param_grid=params_for_grid,
                        scoring=scorer,
                        n_jobs=4,
                        cv = 10)

start = time()
grid_clf = grid_clf.fit(X_train, y_train)
grid_time = time() - start

grid_results[clf_name]['accuracy'] = accuracy_score(grid_clf.predict(X_test), y_test)
grid_results[clf_name]['params'] = grid_clf.best_params_
grid_results[clf_name]['grid time'] = "{} s".format(grid_time)


# select_params = range(2,12)

# # Classifiers
# clfs = [DecisionTreeClassifier(), RandomForestClassifier()]


# # Store results
# for clf in clfs:
#     grid_results = { clf.__class__.__name__ : {} }

# ## Grid Search
# for clf in clfs:
#     clf_name = clf.__class__.__name__
#     steps = [('selector', SelectKBest()),
#              ('classif', clf)]
    
#     params_for_grid = { 'selector__k' : select_params }
#     # print { 'clf__'+key : clf_params[clf_name][key] for key in clf_params[clf_name]}
#     params_for_grid.update( {'classif__'+key : clf_params[clf_name][key] for key in clf_params[clf_name]} )
#     print params_for_grid
    
#     pipeline = Pipeline(steps)
#     scorer = make_scorer(accuracy_score)
    
#     grid_clf = GridSearchCV(clf, param_grid=params_for_grid, scoring=scorer, n_jobs=4, cv = 10)

#     start = time()
#     grid_clf = grid_clf.fit(X_train, y_train)
#     grid_time = time() - start

#     grid_results[clf_name]['accuracy'] = accuracy_score(grid_clf.predict(X_test), y_test)
#     grid_results[clf_name]['params'] = grid_clf.best_params_
#     grid_results[clf_name]['grid time'] = "{} s".format(grid_time)
