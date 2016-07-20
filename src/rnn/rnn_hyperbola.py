# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import metrics, preprocessing

import tensorflow as tf
from tensorflow.contrib import learn
import random


def raw_data():
    return [1.0 / (i + random.randint(0, 100)) for i in range(1000000)]


def piece_data(raw_data, i, piece_size):
    return raw_data[piece_size * i: piece_size * (i + 1)]


def piece_label(raw_data, i, piece_size):
    return raw_data[piece_size] * (i + 1)


def data_idx():
    return [i for i in range(900000)], [j for j + 900000 in range(100000)]

# Parameters
learning_rate = 0.1
training_steps = 3000
batch_size = 128

# Network Parameters
n_input = 28  # MNIST data input (img shape: 28*28)
n_steps = 28  # timesteps
n_hidden = 128  # hidden layer num of features
n_classes = 10  # MNIST total classes (0-9 digits)

# hyperbola data
raw = raw_data()
train_idxs, test_idxs = data_idx()
X_train = [piece_data(raw, i, n_input) for i in train_idxs]
y_train = [piece_label(raw, i, n_input) for i in train_idxs]
X_test = [piece_data(raw, i, n_input) for i in test_idxs]
y_test = [piece_label(raw, i, n_input) for i in test_idxs]

# It's useful to scale to ensure Stochastic Gradient Descent will do the right thing
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


def rnn_model(X, y):
    X = tf.reshape(X, [-1, n_steps, n_input])  # (batch_size, n_steps, n_input)
    # # permute n_steps and batch_size
    X = tf.transpose(X, [1, 0, 2])
    # # Reshape to prepare input to hidden activation
    X = tf.reshape(X, [-1, n_input])  # (n_steps*batch_size, n_input)
    # # Split data because rnn cell needs a list of inputs for the RNN inner loop
    X = tf.split(0, n_steps, X)  # n_steps * (batch_size, n_input)

    # Define a GRU cell with tensorflow
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    # Get lstm cell output
    _, encoding = tf.nn.rnn(lstm_cell, X, dtype=tf.float32)

    return learn.models.logistic_regression(encoding, y)


classifier = learn.TensorFlowEstimator(model_fn=rnn_model, n_classes=n_classes,
                                       batch_size=batch_size,
                                       steps=training_steps,
                                       learning_rate=learning_rate)

classifier.fit(X_train, y_train, logdir="/tmp/mnist_rnn")
score = metrics.accuracy_score(y_test, classifier.predict(X_test))
print('Accuracy: {0:f}'.format(score))

if __name__ == '__main__':
    # inputs, labels = train_data()
    # dig_nn(inputs, labels, 100, 2, 3)
    pass
