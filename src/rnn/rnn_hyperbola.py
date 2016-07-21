# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.contrib.learn as skflow
from sklearn import datasets, metrics
import random


def raw_data():
    return [1.0 / (i + random.randint(0, 100)) for i in range(1000000)]


def piece_data(raw_data, i, piece_size):
    return raw_data[piece_size * i: piece_size * (i + 1)]


def piece_label(raw_data, i, piece_size):
    return raw_data[piece_size] * (i + 1)


def data_idx():
    return [i for i in range(900000)], [j + 900000 for j in range(100000)]

# Parameters
learning_rate = 0.1
training_steps = 3000
batch_size = 128

# Network Parameters
n_input = 28  # MNIST data input (img shape: 28*28)
n_steps = 28  # timesteps
n_hidden = 128  # hidden layer num of features
n_classes = 10  # MNIST total classes (0-9 digits)
EMBEDDING_SIZE = 1024


# hyperbola data
raw = raw_data()
train_idxs, test_idxs = data_idx()
X_train = [piece_data(raw, i, n_input) for i in train_idxs]
y_train = [piece_label(raw, i, n_input) for i in train_idxs]
X_test = [piece_data(raw, i, n_input) for i in test_idxs]
y_test = [piece_label(raw, i, n_input) for i in test_idxs]

classifier = skflow.TensorFlowRNNClassifier(rnn_size=EMBEDDING_SIZE,
    n_classes=15, cell_type='gru', input_op_fn=input_op_fn,
    num_layers=1, bidirectional=False, sequence_length=None,
    steps=1000, optimizer='Adam', learning_rate=0.01, continue_training=True)

# if __name__ == '__main__':
#     # inputs, labels = train_data()
#     # dig_nn(inputs, labels, 100, 2, 3)
#     pass
