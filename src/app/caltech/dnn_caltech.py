from __future__ import print_function

import numpy as np
import tensorflow as tf

from app.caltech.data import read_caltech
from neural.full_connect import accuracy


def tf_deep_nn(regular=False, drop_out=False, lrd=False, layer_cnt=2):
    batch_size = 128

    graph = tf.Graph()
    with graph.as_default():
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, feature_dim))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        hidden_node_count = 32
        # start weight
        hidden_stddev = np.sqrt(2.0 / 100)
        weights1 = tf.Variable(tf.truncated_normal([feature_dim, hidden_node_count], stddev=hidden_stddev))
        biases1 = tf.Variable(tf.zeros([hidden_node_count]))
        # middle weight
        weights = []
        biases = []
        hidden_cur_cnt = hidden_node_count
        for i in range(layer_cnt - 2):
            if hidden_cur_cnt > 2:
                hidden_next_cnt = int(hidden_cur_cnt / 2)
            else:
                hidden_next_cnt = 2
            hidden_stddev = np.sqrt(2.0 / hidden_cur_cnt / 10)
            weights.append(tf.Variable(tf.truncated_normal([hidden_cur_cnt, hidden_next_cnt], stddev=hidden_stddev)))
            biases.append(tf.Variable(tf.zeros([hidden_next_cnt])))
            hidden_cur_cnt = hidden_next_cnt
        # first wx + b
        y0 = tf.matmul(tf_train_dataset, weights1) + biases1
        # first sigmoid
        hidden = tf.nn.sigmoid(y0)
        # hidden = y0
        hidden_drop = hidden
        # first DropOut
        keep_prob = 0.5
        if drop_out:
            hidden_drop = tf.nn.dropout(hidden, keep_prob)
        # first wx+b for valid
        valid_y0 = tf.matmul(tf_valid_dataset, weights1) + biases1
        valid_hidden = tf.nn.sigmoid(valid_y0)
        # valid_hidden = valid_y0
        # first wx+b for test
        test_y0 = tf.matmul(tf_test_dataset, weights1) + biases1
        test_hidden = tf.nn.sigmoid(test_y0)
        # test_hidden = test_y0

        # middle layer
        for i in range(layer_cnt - 2):
            y1 = tf.matmul(hidden_drop, weights[i]) + biases[i]
            hidden_drop = tf.nn.sigmoid(y1)
            if drop_out:
                keep_prob += 0.5 * i / (layer_cnt + 1)
                hidden_drop = tf.nn.dropout(hidden_drop, keep_prob)

            y0 = tf.matmul(hidden, weights[i]) + biases[i]
            hidden = tf.nn.sigmoid(y0)
            # hidden = y0

            valid_y0 = tf.matmul(valid_hidden, weights[i]) + biases[i]
            valid_hidden = tf.nn.sigmoid(valid_y0)
            # valid_hidden = valid_y0

            test_y0 = tf.matmul(test_hidden, weights[i]) + biases[i]
            test_hidden = tf.nn.sigmoid(test_y0)
            # test_hidden = test_y0

        # last weight
        weights2 = tf.Variable(tf.truncated_normal([hidden_cur_cnt, num_labels], stddev=hidden_stddev / 2))
        biases2 = tf.Variable(tf.zeros([num_labels]))
        # last wx + b
        logits = tf.matmul(hidden_drop, weights2) + biases2

        # predicts
        logits_predict = tf.matmul(hidden, weights2) + biases2
        valid_predict = tf.matmul(valid_hidden, weights2) + biases2
        test_predict = tf.matmul(test_hidden, weights2) + biases2

        l2_loss = 0
        # enable regularization
        if regular:
            l2_loss = tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2)
            for i in range(len(weights)):
                l2_loss += tf.nn.l2_loss(weights[i])
                # l2_loss += tf.nn.l2_loss(biases[i])

            beta = 1e-2
            l2_loss *= beta
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels)) + l2_loss

        # Optimizer.
        if lrd:
            cur_step = tf.Variable(0, trainable=False)  # count the number of steps taken.
            starter_learning_rate = 0.4
            learning_rate = tf.train.exponential_decay(starter_learning_rate, cur_step, 500, 0.75, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=cur_step)
        else:
            optimizer = tf.train.AdamOptimizer(0.5).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits_predict)
        valid_prediction = tf.nn.softmax(valid_predict)
        test_prediction = tf.nn.softmax(test_predict)

    num_steps = 8001

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print("Initialized")
        for step in range(num_steps):
            offset_range = train_labels.shape[0] - batch_size
            offset = (step * batch_size) % offset_range
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if step % 50 == 0:
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(
                    valid_prediction.eval(), valid_labels))
        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


if __name__ == '__main__':
    feature_dim = 2330
    num_labels = 2
    train_dataset, train_labels, test_dataset, test_labels = read_caltech()
    valid_dataset = train_dataset[:99]
    valid_labels = train_labels[:99]
    train_dataset = np.asarray(train_dataset, dtype=np.float32)
    train_labels = np.asarray(train_labels, dtype=np.int32)
    valid_dataset = np.asarray(valid_dataset, dtype=np.float32)
    valid_labels = np.asarray(valid_labels, dtype=np.int32)
    test_dataset = np.asarray(test_dataset, dtype=np.float32)
    test_labels = np.asarray(test_labels, dtype=np.int32)
    tf_deep_nn(layer_cnt=3, lrd=True, drop_out=False, regular=True)
