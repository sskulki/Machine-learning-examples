#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 21:48:29 2018

@author: shridhar
"""

#Install Theano
#pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

#Install TensorFlow
#Install TensorFlow from website: http://www.tensorflow.org/versions/r0.11/get_started/os.setup.html
#conda install -c anaconda tensorflow 

#Install Keras
#pip install --upgrade keras

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

#Encoding categorical independent variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_1.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X = X[:,1:]
#Splitting the dataset into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
#Add input and first hidden layer
classifier.add(Dense(output_dim = 6, init='uniform', activation='relu', input_dim=11))
#Add second hidden layer
classifier.add(Dense(output_dim = 6, init='uniform', activation='relu'))
#Add output layer
classifier.add(Dense(output_dim = 1, init='uniform', activation='sigmoid'))
#Compile
classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
#Fit and train the ANN
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

#Predict the test series
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)

#86% accuracy without tuning the parameters





















