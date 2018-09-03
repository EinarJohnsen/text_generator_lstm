'''
Initial Code is taken from and from and Jason Brownlee:

https://github.com/bhaveshoswal/CNN-text-classification-keras/blob/master/model.py

https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

Note that the buildModel function is not necessarily the same as in the experiments.
Use the weights file to recreate the results if needed.

'''
from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.utils.data_utils import get_file
from keras.models import load_model
from keras.layers.wrappers import TimeDistributed
from keras.layers import Embedding, Merge
from keras.datasets import imdb
from keras.preprocessing import sequence
import numpy as np
import random
import sys
import os

model_file = "weights/lstm_classify.h5"

(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words=None,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3,
                                                      nb_words=5000)
top_words = 5000
embedding_vecor_length = 64
max_review_length = 300
number_elements = 25000

X_train = sequence.pad_sequences(x_train, maxlen=max_review_length)[0: number_elements]
X_test = sequence.pad_sequences(x_test, maxlen=max_review_length)[0: number_elements]

y_train = y_train[0: number_elements]
y_test = y_test[0: number_elements]


def buildModel(x_train, y_train, model_file, batch_size=128, epochs=100):

    model = Sequential()
    model.add(Embedding(5000, 64, input_length=max_review_length, name="emb1"))
    model.add(LSTM(512))
    model.add(Dropout(0.3, name="d1"))
    model.add(Dense(128, name="d2", activation="relu"))
    model.add(Dense(1, activation='sigmoid', name="m_d1"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    model.save(model_file)


def train_current_Model(x_train, x_test, my_model, batch_size, epochs=1):
    model = load_model(my_model)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    model.save(my_model)


def testModel(x_test, y_test):
    model = load_model(model_file)
    scores = model.evaluate(x_test, y_test, verbose=1)
    print(scores)
    avalue = model.predict([x_test])

    score = 0
    for i in range(0, 25000):
        value = avalue[i][0]
        print(value, y_test[i])
        if(value < 0.5 and y_test[i] == 0):
            score += 1
        if(value >= 0.5 and y_test[i] == 1):
            score += 1

    print(score/25000)


def showSummary():
    model = load_model(model_file)
    print(model.summary())

# buildModel(X_train, y_train, model_file, batch_size=256, epochs=1)
# train_current_Model(x_train, x_test, model_file, batch_size=1, epochs=1)
# testModel(X_test, y_test)
# showSummary()
