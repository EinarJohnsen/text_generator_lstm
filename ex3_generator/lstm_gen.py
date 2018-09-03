'''
Initial LSTM code and file reading structure is taken from: Jason Brownlee at machinelearningmastery.com

https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

Note that the buildModel function is not necessarily the same as in the experiments.
Use the weights file to recreate the results if needed.

'''

import numpy as numpy
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout, Flatten
from keras.layers import LSTM, RNN
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.utils.data_utils import get_file
from keras.layers import Embedding, Merge
from keras.preprocessing import sequence

filename = "data/alice_better.txt"
raw_text = open(filename, encoding="utf-8").read()
raw_text = raw_text.lower()
raw_text = raw_text


chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

seq_length = 100
dataX = []
dataY = []

for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

X = numpy.reshape(dataX, (n_patterns, seq_length))
y = np_utils.to_categorical(dataY)
X = sequence.pad_sequences(X, maxlen=100)


def buildModel(X, y, epochs=1, batch_size=128):
    model = Sequential()
    model.add(Embedding(n_vocab, 64, input_length=100, batch_input_shape=(1, 100), name="embii1"))
    model.add(LSTM(64, input_shape=(n_patterns, 64), batch_input_shape=(1, 100, 64), return_sequences=False, name="sfsas", stateful=True))
    model.add(Dropout(0.2, name="D1"))
    model.add(Dense(y.shape[1], activation='softmax', name="D2"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    filepath = "lt-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)


def evaluateModel(filepath, X):
    model = load_model(filepath)
    start = numpy.random.randint(0, len(dataX)-1)
    pattern = numpy.reshape(X[start], (1, 100))
    seed = ""
    for x in X[start]:
        seed += int_to_char[x]
    print(seed)
    my_result = ""
    for i in range(100):
        prediction = model.predict(pattern, verbose=0)
        index = numpy.argmax(prediction) 
        result = int_to_char[index]
        a_pattern = pattern[0].tolist()
        a_pattern.append(index)
        a_pattern = a_pattern[1:len(a_pattern)] 
        pattern = numpy.reshape(a_pattern, (1, 100))
        my_result += result
    print("\nDone.")
    print(my_result, " <--- result")
