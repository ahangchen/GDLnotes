# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.contrib.learn as skflow
import numpy as np
import random
import tensorflow as tf


def raw_data():
    return [1.0 / (i + 1) for i in range(1000000)]


def piece_data(raw_data, i, piece_size):
    return raw_data[piece_size * i: piece_size * (i + 1)]


def piece_label(raw_data, i, piece_size):
    return raw_data[piece_size * (i + 1)]


def data_idx():
    # return [i for i in range(9000)], [j + 9000 for j in range(1000)]
    return [i for i in range(900)], [j + 90 for j in range(100)]


# Parameters
learning_rate = 0.1
training_steps = 3000

# Network Parameters
batch_size = 100  # MNIST data input (img shape: 28*28)
p_size = 10  # 10 num to predict one num
n_hidden = 128  # hidden layer num of features
n_classes = 10  # MNIST total classes (0-9 digits)
EMBEDDING_SIZE = 1024

# hyperbola data
raw = raw_data()

train_idxs, test_idxs = data_idx()
X_train = np.array([piece_data(raw, i, p_size) for i in train_idxs])
y_train = np.array([piece_label(raw, i, p_size) for i in train_idxs])
X_test = np.array([piece_data(raw, i, p_size) for i in test_idxs])
y_test = np.array([piece_label(raw, i, p_size) for i in test_idxs])


def x2y(x):
    print(x)
    return tf.split(1, p_size, x)

classifier = skflow.TensorFlowRNNClassifier(rnn_size=EMBEDDING_SIZE,
                                            n_classes=3, cell_type='gru', input_op_fn=x2y,
                                            num_layers=2, bidirectional=False, sequence_length=None,
                                            steps=1, optimizer='Adam', learning_rate=0.01, continue_training=True)

classifier.fit(X_train, y_train)
y_predict = classifier.predict(X_test)
print(y_predict)
print(y_test)

