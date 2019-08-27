from __future__ import print_function
import collections
import os
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras import backend as K
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pickle



def read_words(filename):
    with open(filename) as f:
        return f.read().replace("\n", "<eos>").lower().split()


def build_vocab(filename):
    data = read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def file_to_word_ids(filename, word_to_id):
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def load_data():


    data_path = "data/"
    train_path = data_path + "ptb.train.txt"
    valid_path = data_path + "ptb.valid.txt"
    test_path = data_path + "ptb.test.txt"

    # build the complete vocabulary, then convert text data to list of integers
    word_to_id = build_vocab(train_path)
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))

    return train_data, valid_data, test_data, vocabulary, reversed_dictionary

train_data, valid_data, test_data, vocabulary, reversed_dictionary = load_data()


class KerasBatchGenerator(object):

    def __init__(self, data, num_steps, batch_size, vocabulary, skip_step=5):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        # skip_step is the number of words which will be skipped before the next
        self.skip_step = skip_step

    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.num_steps, self.vocabulary))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
                temp_y = self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]
                # convert all of temp_y into a one hot representation
                y[i, :, :] = to_categorical(temp_y, num_classes=self.vocabulary)
                self.current_idx += self.skip_step
            yield x, y


num_steps = 32
batch_size = 32
train_data_generator = KerasBatchGenerator(train_data, num_steps, batch_size, vocabulary,
                                           skip_step=num_steps)
valid_data_generator = KerasBatchGenerator(valid_data, num_steps, batch_size, vocabulary,
                                           skip_step=num_steps)


def perplexity(y_true, y_pred):
    return np.exp(K.mean(K.categorical_crossentropy(y_true, y_pred)))

hidden_size = 300
use_dropout=True
model = Sequential()
model.add(Embedding(vocabulary, hidden_size, input_length=num_steps))
model.add(LSTM(hidden_size, return_sequences=True))
if use_dropout:
    model.add(Dropout(0.5))
model.add(LSTM(hidden_size, return_sequences=True))
if use_dropout:
    model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(vocabulary)))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy', perplexity])

print(model.summary())
checkpointer = ModelCheckpoint(filepath=data_path + 'final_run/model-{epoch:02d}.hdf5', verbose=1)

#print("loading epoch 19 saved model")
#model.load_weights(data_path+"/model-19.hdf5")

num_epochs = 50
callback_history = model.fit_generator(train_data_generator.generate(), len(train_data)//(batch_size*num_steps), num_epochs,
                        validation_data=valid_data_generator.generate(),
                        validation_steps=len(valid_data)//(batch_size*num_steps), callbacks=[checkpointer])

model.save(data_path + "final_run/final_model.hdf5")
with open(data_path+'final_run/trainHistoryDict', 'wb') as file_pi:
        pickle.dump(callback_history.history, file_pi)
 
plt.plot(callback_history.history['perplexity'])
plt.plot(callback_history.history['val_perplexity'])
plt.title('Model Perplexity')
plt.ylabel('Perplexity')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig('nn_4_50.png')

model = load_model(data_path + "/final_run/model-50.hdf5", custom_objects={'perplexity':perplexity})

#model = load_model("model-10.hdf5")

def predict(data, num_predict=300):
    generated = ''
    start_index = random.randint(0, len(data) - num_steps - 1)
    # finding seed data by randomly selecting an index
    sequence = data[start_index: start_index + num_steps]
    for i in sequence:
        generated += reversed_dictionary[i]
        
    sequence = np.array([sequence])
    print('----- Generating with seed: "' + generated + '"')
    print()
    
    seq = sequence
    for i in range(num_predict):

        predictions = model.predict(seq)
        
        predict_word = np.argmax(predictions[:, num_steps-1, :])
        generated += reversed_dictionary[predict_word] + " "
        
        seq = np.array([np.append(seq[0][1:], [predicted_id])])
        
    return generated
	
print(predict(test_data))
	


