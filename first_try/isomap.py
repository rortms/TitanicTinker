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
from sklearn.neighbors import KNeighborsClassifier

## Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

## Tuning
from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

## Pipe
from sklearn.pipeline import Pipeline


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

######################################
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

from sklearn.manifold import Isomap
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

scaler = MinMaxScaler()
uX_all = scaler.fit_transform(X_all)

ismap = Isomap(n_neighbors=4)
pX_all = ismap.fit_transform(uX_all)

survived = np.array(
    [ pX_all[i] for i in range(pX_all.shape[0]) if y_all.iloc[i] == 1]
)

died = np.array(
    [ pX_all[i] for i in range(pX_all.shape[0]) if y_all.iloc[i] == 0]
)

plt.subplot(1, 2, 1)
plt.plot(survived[:,0],survived[:,1], 'bo')
plt.subplot(1, 2, 2)
plt.plot(died[:,0], died[:,1], 'ro')
# plt.show()

# ##########################
# # Train/Test Split
# fX_all = np.concatenate((uX_all, pX_all), 1)
X_train, X_test, y_train, y_test = train_test_split(pX_all, y_all,
                                                    test_size= 0.25,
                                                    random_state = 123)


# survived = np.array(
#     [ X_train[i] for i in range(X_train.shape[0]) if y_train.iloc[i] == 1]
# )

# died = np.array(
#     [ X_train[i] for i in range(X_train.shape[0]) if y_train.iloc[i] == 0]
# )

# plt.subplot('121')
# plt.plot(survived[:,0],survived[:,1], 'bo')
# plt.subplot('122')
# plt.plot(died[:,0], died[:,1], 'ro')
# plt.show()

clf = KNeighborsClassifier(n_neighbors=4, algorithm='brute')

start = time()
clf = clf.fit(X_train, y_train)
grid_time = time() - start

print "Acc on Train: ", accuracy_score(clf.predict(X_train), y_train)
print "Acc on Test: ", accuracy_score(clf.predict(X_test), y_test)

# ####################
# # Gridsearch Tuning

# grid_results = {}
# params =  dict( KBest__k=range(6,14),
                
#                 # Random Forest Params
#                 ran_forest__n_estimators = range(3,7),
#                 ran_forest__criterion= ["entropy", "gini"],
#                 ran_forest__max_features=["sqrt", "log2"],
#                 ran_forest__random_state= [0],)
                
#                 # ran_forest__max_depth= range(3,8),)
#                 # ran_forest__min_samples_split=range(2,8),
#                 # ran_forest__min_samples_leaf=range(2,8) )

# steps = [('KBest', SelectKBest()),
#          ('ran_forest', RandomForestClassifier(random_state=123))]

# for partition in range(2,7):
#     pipe = Pipeline(steps)

#     scorer = make_scorer(accuracy_score)
#     grid_clf = GridSearchCV(pipe, param_grid=params, scoring=scorer,
#                             n_jobs = 4,
#                             cv=partition)
#     start = time()
#     grid_clf = grid_clf.fit(X_train, y_train)
#     grid_time = time() - start

#     train_acc = accuracy_score(grid_clf.predict(X_train), y_train)
    
#     grid_results['accuracy']= accuracy_score(grid_clf.predict(X_test), y_test)
#     grid_results['params'] = grid_clf.best_params_
#     grid_results['grid time'] = "{} s".format(grid_time)

#     print
#     print "{} cv buckets: ".format(partition)
#     print "Training accuracy: {}".format(train_acc)
#     print grid_results['accuracy'], grid_results['grid time'],
#     print 
