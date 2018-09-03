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
from keras.layers import Embedding, Merge, Average, average
from keras.datasets import imdb
from keras.preprocessing import sequence
import numpy as np
import random
import sys
import os
import keras
from keras import backend as K

lstm_path = "lstm_classify.h5"
cnn_path = "cnn_classify.hdf5"

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

model = Sequential()


def testModel(x_test, y_test, model_file, lstm_path, cnn_path):

    test_Model = Sequential()
    lstm_model = load_model(lstm_path)
    cnn_model = load_model(cnn_path)
    merged = Merge([lstm_model, cnn_model], name="hihi", mode='ave')
    test_Model.add(merged)

    test_Model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    avalue = test_Model.predict([x_test, x_test])
    score = 0
    for i in range(0, 25000):
        value = avalue[i][0]
        if(value < 0.5 and y_test[i] == 0):
            score += 1
        if(value >= 0.5 and y_test[i] == 1):
            score += 1

    print(score/25000)


testModel(X_test, y_test, model_file, lstm_path, cnn_path)
