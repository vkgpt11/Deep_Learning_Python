# -*- coding: utf-8 -*-

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# importing the dataset
data = pd.read_csv('Churn_Modelling.csv')
X = data.iloc[:,3:13].values
y = data.iloc[:,13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])

labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])

onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()

X = X[:,1:]

# Spliting the dataset into the training and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Importing keras
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense

#initializing the ANN
classifier = Sequential()

# 1. Randomly initialise the weights to small numbers close to 0

# 2. Input the first observation of your dataset in the input layer, each feature in one input node --> 11

# 3. Forward-propagation: from left to right, the neurons are activated in a way that the impact of each neuron's 
#    Activation is limited by the weights. Propagate the activations until getting the predicted result y

# 4. Compare the predicted result to the actual result. Measure the generated error.

# 5. Back-Propagation: from right to left. the error is back propagated . Update the weights according to 
#   how much they are responsible for the error. the learning rate decides by how much we update the weights

# 6. Repeat Steps 1 to 5 and update the weights after each observation (Reinforcement Learning) Or 
#    Repeat Step 1 to 5 but update the weights only after a batch of observations (Batch Learning).

# 7. When the whole training set passed through the ANN, that makes an epoch. Redo more Epochs


# Adding the input layer and the first hidden layer 
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu',input_dim=11))

# Adding the second hidden layer 
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# Adding the output layer 
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Fitting the ANN to training set
classifier.fit(X_train,y_train,batch_size=10,nb_epoch=100)

# predicting the test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

# making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
accuracy = (cm[0,0]+cm[1,1]) / y_test.shape[0]
accuracy
### Evaluating, Improving and Tuning the ANN -> Variance bias tradeoff
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu',input_dim=11))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier


classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator=classifier, X=X_train,y=y_train,cv=10)
mean = accuracies.mean()
variance = accuracies.std()

# Improving the ANN
# Dropout Regularization to reduce overfitting if needed
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
mean = accuracies.mean()
variance = accuracies.std()

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
#best_params_= {'batch_size': 25, 'epochs': 500, 'optimizer': 'rmsprop'}
import pickle
pickle.dump(grid_search, open('data.pkl', 'wb'))
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


