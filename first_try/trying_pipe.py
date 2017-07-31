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
#data = pd.read_csv("titanic_data.csv")
data = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")


#########################################
# Get features and Dummy if necessary

# Create Cabin boolean feature (Did they have a cabin number or not)
has_cabin = pd.Series([1 if v == False else 0 for v in data['Cabin'].isnull()], )
has_cabin_test = pd.Series([1 if v == False else 0 for v in test['Cabin'].isnull()], )

# All relevant features (Cabin will be added back)
features = [f for f in data.columns if f not in
            ["Survived", 'Name', 'PassengerId', 'Cabin', 'Ticket' ] ]

# Filter features
X_all = pd.DataFrame(data[features], columns=features)
X_test = pd.DataFrame(test[features], columns=features)

# Add binary cabin feature
X_all['Cabin'] = has_cabin
X_test['Cabin'] = has_cabin_test

# Create target vector
y_all = pd.Series(data['Survived'])


## Replace NaN ages with age mean/median
m_age = pd.concat([data['Age'], test['Age']]).median()
X_all['Age'].fillna(m_age, inplace=True)
X_test['Age'].fillna(m_age, inplace=True)
X_test['Fare'].fillna(X_test['Fare'].median(), inplace=True) # Apparently test set has 1 nan at index 152

## Remove datapoints with NaN age
# good_indices = [i for i in X_all.index if not X_all['Age'].isnull()[i]]
# X_all = X_all.iloc[good_indices]
# y_all = np.array(y_all.iloc[good_indices])

## Sanity checks
# print reduce(lambda x,y: x and y, list(X_all['Cabin'] == has_cabin))
# NaNs = sum([1 for v in list(X_all['Cabin'].isnull()) if v == True])
# print 1.*NaNs/ X_all.shape[0] *100

###############################
# Make categorical into dummies
X_all = np.array(cat2Dummies(X_all))
X_test= np.array(cat2Dummies(X_test))
# print X_all.describe()



########
# T-SNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
#
scaler = MinMaxScaler()
uX_all = scaler.fit_transform(X_all)

# Since our data set is not large we can indulge in
# the exact t-sne algorithm instead of the speedier 'barnes-hut'.
# from sklearn.manifold import TSNE
# model = TSNE(n_components=2,
#              early_exaggeration=4.0,
#              learning_rate=1000,
#              n_iter=1000,
#              init='pca',
#              random_state=0,
#              method='exact')
# clusters = model.fit_transform(uX_all)


# survived = np.array([ clusters[i] for i in range(clusters.shape[0]) if y_all[i] == 1 ])
# died = np.array([ clusters[i] for i in range(clusters.shape[0]) if y_all[i] == 0 ] )

# plt.plot(survived[:,0],survived[:,1], 'bo')
# plt.plot(died[:,0], died[:,1], 'ro')
# plt.show()
# # plt.savefig("./figures/t-sne_all.png")


############################
# Train/Test Split
X_train, X_cv, y_train, y_cv = train_test_split(X_all, y_all,
                                                    test_size= 0.25,
                                                    random_state = 123)

# ENd Pre-process
###

##########################################
# Preliminary Pipe KBest/RandomForest
print
print 'Preliminary Pipe KBest/RandomForest'
best_feats, max_acc = None, 0
for i in range(1,11):
    clf = RandomForestClassifier(random_state=123)
    select = SelectKBest(k=i)
    steps = [('KBest', select),
             ('random_forest', clf)]
    
    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)
    
    y_prediction = pipeline.predict(X_cv)
    acc_score = accuracy_score(y_prediction, y_cv)
    
    if acc_score > max_acc:
        best_feats = select.get_support(indices=True)
    print "{} best features accuracy: {}".format(i,  acc_score)
    #, [X_train.columns[k] for k in select.get_support(indices=True)]


    
# ## Decision Tree    
# ## 3 best features accuracy: 0.793296089385
# ['Pclass', 'Sex_female', 'Sex_male']
# ## 9 best features accuracy: 0.798882681564
# ['Pclass', 'Sex_female', 'Sex_male', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Cabin']


# ## Forest 
# ## 10 best features accuracy: 0.826815642458
# ['Pclass', 'Sex_female', 'Sex_male', 'Age', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Cabin']


##########################################
# Preliminary Pipe KBest/RandomForest
print 
print "Preliminary Pipe PCA/RandomForest "
# best_feats, max_acc = None, 0
for i in range(1,11):
    clf = RandomForestClassifier(random_state=123)
    select = PCA(n_components=i, whiten=True, random_state=123)
    
    steps = [('PCA', select),
              ('random_forest', clf)]
    
    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)
    
    y_prediction = pipeline.predict(X_cv)
    acc_score = accuracy_score(y_prediction, y_cv)
    
    # if acc_score > max_acc:
    #     best_feats = select.get_support(indices=True)
    print "{} components: {}".format(
        i,
        acc_score)#, [X_train.columns[k] for k in select.get_support(indices=True)]
    
####################
# Gridsearch Tuning
grid_results = {}
params =  dict( KBest__k=range(6,11),
                
                # Random Forest Params 
                ran_forest__n_estimators = range(3,7),
                ran_forest__criterion= ["entropy", "gini"],
                ran_forest__max_features=["sqrt", "log2"],
                ran_forest__random_state= [0],)
                
                # ran_forest__max_depth= range(3,8),)
                # ran_forest__min_samples_split=range(2,8),
                # ran_forest__min_samples_leaf=range(2,8) )

steps = [('KBest', SelectKBest()),
         ('ran_forest', RandomForestClassifier(random_state=123))]

#########################################
# Searching for good CV number of buckets
# for partition in range(2, 6):

#     pipe = Pipeline(steps)

#     scorer = make_scorer(accuracy_score)
#     grid_clf = GridSearchCV(pipe, param_grid=params, scoring=scorer,
#                             n_jobs = 4,
#                             cv=partition)
#     start = time()
#     grid_clf = grid_clf.fit(X_train, y_train)
#     grid_time = time() - start

#     train_acc = accuracy_score(grid_clf.predict(X_train), y_train)
    
#     grid_results['accuracy']= accuracy_score(grid_clf.predict(X_cv), y_cv)
#     grid_results['params'] = grid_clf.best_params_
#     grid_results['grid time'] = "{} s".format(grid_time)

#     print
#     print "{} cv buckets: ".format(partition)
#     print "Training accuracy: {}".format(train_acc)
#     print grid_results['accuracy'], grid_results['grid time'], 
#     print


# ########################
# # Fiting and Predicting
pipe = Pipeline(steps)
partition = 4
scorer = make_scorer(accuracy_score)

grid_clf = GridSearchCV(pipe, param_grid=params, scoring=scorer,
                        n_jobs = 4,
                        cv=partition)
start = time()
grid_clf = grid_clf.fit(X_train, y_train)
grid_time = time() - start

train_acc = accuracy_score(grid_clf.predict(X_train), y_train)

grid_results['accuracy']= accuracy_score(grid_clf.predict(X_cv), y_cv)
grid_results['params'] = grid_clf.best_params_
grid_results['grid time'] = "{} s".format(grid_time)

print
print "{} cv buckets: ".format(partition)
print "Training accuracy: {}".format(train_acc)
print "Cross-val accuracy: {}".format(grid_results['accuracy'])
print "Training time: {}".format(grid_results['grid time'])
print

# Create prediction csv
predictions = grid_clf.predict(X_test)
np.savetxt("./random_forest_prediction.csv", predictions, delimiter=',')




