import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

import keras


######### Change to Theano Backend ########
from keras.utils import to_categorical
from keras import backend as K
def set_keras_backend(backend):

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend

set_keras_backend("theano")
###########################################


#################
# Helper function 
def dummify(X):            # Make dummies for string values
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


# data = pd.read_csv("titanic_data.csv")
data = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

##############
# Get features

# Create Cabin boolean feature (Did they have a cabin number or not)
has_cabin = pd.Series([1 if v == False else 0 for v in data['Cabin'].isnull()], )

# All relevant features (Cabin will be added back)
features = [f for f in data.columns if f not in
            ["Survived", 'Name', 'PassengerId', 'Cabin', 'Ticket' ] ]

X_all = pd.DataFrame(data[features], columns=features)

X_all['Cabin'] = has_cabin
y_all = pd.get_dummies(pd.Series(data['Survived']))

good_indices = [i for i in X_all.index if not X_all['Age'].isnull()[i]]
X_all = X_all.iloc[good_indices]
y_all = np.array(y_all.iloc[good_indices])


## Sanity checks
# print reduce(lambda x,y: x and y, list(X_all['Cabin'] == has_cabin))
# NaNs = sum([1 for v in list(X_all['Cabin'].isnull()) if v == True])
# print 1.*NaNs/ X_all.shape[0] *100

###############################
# Make categorical into dummies
X_all = np.array(dummify(X_all))
# print X_all.describe()

from sklearn.model_selection import train_test_split
predictors, X_test, target, y_test = train_test_split(X_all, y_all,
                                                    test_size= 0.25,
                                                    random_state = 123)


###############
## KERAS MODEL

np.random.seed(123)
random.seed(123)

from keras.models import Sequential
from keras.layers import Dense, Dropout

from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.metrics import binary_accuracy

for k in [3,30,50]:
    np.random.seed(123)
    random.seed(123)
    
    model = Sequential()
    model.add(Dense(128, activation='relu',input_shape=(X_all.shape[1],)))
    model.add(Dense(2, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping_monitor = EarlyStopping(patience=2)
    model.fit(predictors, target,
              validation_split = 0.3, epochs = k,
              verbose=0,
              callbacks = [early_stopping_monitor])
    # model.fit(X_train, y_train, verbose=0)

    train_loss, train_acc = model.evaluate(predictors, target)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    
    print '\nepochs:\t{}'.format(k)
    print 'Train loss:\t{}'.format(train_loss)
    print 'Train accuracy:\t{}'.format(train_acc)
    print 
    print 'Test loss:\t{}'.format(test_loss)
    print 'Test accuracy:\t{}'.format(test_acc)                                           
    # tf.reset_default_graph()


# from keras.callbacks import EarlyStopping

# early_stopping_monitor = EarlyStopping(patience=2)
# model.fit(predictors, target, validation_split = 0.3, epochs = 20,
#           callbacks = [early_stopping_monitor])

# def get_new_model(input_shape = input_shape):
#     model = Sequential()
#     model.add(Dense(100, activation='relu', input_shape = input_shape))
#     model.add(Dense(100, activation='relu'))
#     model.add(Dense(2, activation='softmax'))
#     return(model)


