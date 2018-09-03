'''
Initial LSTM code and file reading structure is taken from: Jason Brownlee at machinelearningmastery.com

https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

'''
import numpy
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout, Conv1D, Flatten
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.utils.data_utils import get_file


# load ascii text and covert to lowercase
filename = "data/alice_better.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
raw_text = raw_text


# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

# prepare the dataset of input to output pairs encoded as integers
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

# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)


print(X.shape[1], X.shape[2])

def buildModel(X, y, epochs=1, batch_size=128):

    model = Sequential()
    model.add(Conv1D(64, 3, border_mode='same', input_shape=(X.shape[1], X.shape[2])))
    model.add(Flatten())
    model.add(Dropout(0.4, name="D1"))
    model.add(Dense(128, activation='relu', name="intermediate_output"))
    model.add(Dense(y.shape[1], activation='softmax', name="S1"))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    filepath = "weights-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)


def evaluateModel(filepath, X):
    model = load_model(filepath)
    start = numpy.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    print("Seed:")
    print("\"", ''.join([int_to_char[value] for value in pattern]), "\"", " is the value")
    my_result = ""
    # generate characters
    for i in range(50):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
        my_result += result
    print("\nDone.")
    print(my_result, " <--- result")

# buildModel(X, y, 45, 64)
# evaluateModel("weights/weights-15-2.5741.hdf5", X)
# evaluateModel("weights/weights-45-2.8962.hdf5", X)
