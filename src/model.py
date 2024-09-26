## Define the model

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def build_model(input_dim):
    clf = Sequential()
    clf.add(Dense(256, activation='relu', input_dim=input_dim))
    clf.add(Dense(128, activation='relu'))
    clf.add(Dense(64, activation='relu'))
    clf.add(Dense(1, activation='sigmoid'))
    
    clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return clf
