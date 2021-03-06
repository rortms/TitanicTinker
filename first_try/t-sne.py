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


#######################################
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


########
# T-SNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
#
scaler = MinMaxScaler()
uX_all = scaler.fit_transform(X_all)

# Since our data set is not large we can indulge in
# the exact t-sne algorithm instead of the speedier 'barnes-hut'.
from sklearn.manifold import TSNE

model = TSNE(n_components=2,
             early_exaggeration=4.0,
             learning_rate=1000,
             n_iter=1000,
             init='pca',
             random_state=0,
             method='exact')
pX_all = model.fit_transform(uX_all)

survived = np.array([ pX_all[i] for i in range(pX_all.shape[0]) if y_all.iloc[i] == 1])
died = np.array([ pX_all[i] for i in range(pX_all.shape[0]) if y_all.iloc[i] == 0] )

plt.subplot('121')
plt.plot(survived[:,0],survived[:,1], 'bo')
plt.subplot('122')
plt.plot(died[:,0], died[:,1], 'ro')
plt.show()

print "Done"
