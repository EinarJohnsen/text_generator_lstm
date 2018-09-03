'''
Initial LSTM code and file reading structure is taken from: Jason Brownlee at machinelearningmastery.com

https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

Note that the buildModel function is not necessarily the same as in the experiments.
Use the weights file to recreate the results if needed.

'''
import numpy
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, Merge
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.utils.data_utils import get_file

filename = "data/alice_better.txt"
raw_text = open(filename, encoding='utf-8').read()
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

X = numpy.reshape(dataX, (n_patterns, seq_length, 1))

X = X / float(n_vocab)
y = np_utils.to_categorical(dataY)


def buildModel(X, y, epochs=1, batch_size=128):

    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)


def evaluateModel(filepath_lstm, filepath_cnn, X, i):
    lstm_model = load_model(filepath_lstm)
    cnn_model = load_model(filepath_cnn)
    model = Sequential()
    start = numpy.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    merged = Merge([lstm_model, cnn_model], mode="ave")
    model.add(merged)
    print("Seed:")
    print("\"", ''.join([int_to_char[value] for value in pattern]), "\"", " is the value")
    my_result = ""
    for i in range(50):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict([x, x], verbose=0)
        index = numpy.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
        my_result += result
    print("\nDone.")
    print(my_result, " <--- result")


def showSummary(path1, path2):
    a = load_model(path1)
    b = load_model(path2)
    print(a.summary())
    print(b.summary())


# evaluateModel("weights/weights-improvement-45-1.4518.hdf5", "weights/weights-45-2.8962.hdf5", X, 1)
# showSummary("weights/weights-improvement-45-1.4518.hdf5", "weights/weights-45-2.8962.hdf5")
