from __future__ import print_function

import tensorflow as tf
import numpy as np

from neural.full_connect import load_reformat_not_mnist, accuracy


def tf_better_nn(offset_range=-1, regular=False, drop_out=False, lrd=False):
    batch_size = 128

    graph = tf.Graph()
    with graph.as_default():
        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(tf.float32,
                                          shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        hidden_node_count = 1024
        # Variables.
        weights1 = tf.Variable(
            tf.truncated_normal([image_size * image_size, hidden_node_count]))
        biases1 = tf.Variable(tf.zeros([hidden_node_count]))

        weights2 = tf.Variable(
            tf.truncated_normal([hidden_node_count, num_labels]))
        biases2 = tf.Variable(tf.zeros([num_labels]))

        # Training computation. right most
        ys = tf.matmul(tf_train_dataset, weights1) + biases1
        hidden = tf.nn.relu(ys)
        h_fc = hidden

        valid_y0 = tf.matmul(tf_valid_dataset, weights1) + biases1
        valid_hidden1 = tf.nn.relu(valid_y0)

        test_y0 = tf.matmul(tf_test_dataset, weights1) + biases1
        test_hidden1 = tf.nn.relu(test_y0)

        # enable DropOut
        keep_prob = tf.placeholder(tf.float32)
        if drop_out:
            hidden_drop = tf.nn.dropout(hidden, keep_prob)
            h_fc = hidden_drop

        # left most
        logits = tf.matmul(h_fc, weights2) + biases2
        # only drop out when train
        logits_predict = tf.matmul(hidden, weights2) + biases2
        valid_predict = tf.matmul(valid_hidden1, weights2) + biases2
        test_predict = tf.matmul(test_hidden1, weights2) + biases2
        # loss
        l2_loss = tf.nn.l2_loss(weights1) + tf.nn.l2_loss(biases1) + tf.nn.l2_loss(weights2) + tf.nn.l2_loss(biases2)
        # enable regularization
        if not regular:
            l2_loss = 0
        beta = 0.002
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) + beta * l2_loss

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
        if lrd:
            cur_step = tf.Variable(0)  # count the number of steps taken.
            starter_learning_rate = 0.1
            learning_rate = tf.train.exponential_decay(starter_learning_rate, cur_step, 10000, 0.96, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=cur_step)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits_predict)
        valid_prediction = tf.nn.softmax(valid_predict)
        test_prediction = tf.nn.softmax(test_predict)

    num_steps = 3001

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print("Initialized")
        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            if offset_range == -1:
                offset_range = train_labels.shape[0] - batch_size

            offset = (step * batch_size) % offset_range
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, keep_prob: 0.5}
            _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if step % 500 == 0:
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(
                    valid_prediction.eval(), valid_labels))
        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


def tf_deep_nn(offset_range=-1, regular=False, drop_out=False, lrd=False, layer_cnt=2):
    batch_size = 256

    graph = tf.Graph()
    with graph.as_default():
        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(tf.float32,
                                          shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        hidden_node_count = 1024
        # Variables.
        hidden_stddev = np.sqrt(1.0 / 784)
        weights1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_node_count], stddev=hidden_stddev))
        biases1 = tf.Variable(tf.zeros([hidden_node_count]))

        weights = []
        biases = []
        hidden_cur_cnt = hidden_node_count
        for i in range(layer_cnt - 2):
            if hidden_cur_cnt > 2:
                hidden_next_cnt = int(hidden_cur_cnt / 2)
            else:
                hidden_next_cnt = 2
            hidden_stddev = np.sqrt(1.0 / hidden_cur_cnt)
            weights.append(tf.truncated_normal([hidden_cur_cnt, hidden_next_cnt], stddev=hidden_stddev))
            biases.append(tf.Variable(tf.zeros([hidden_next_cnt])))
            hidden_cur_cnt = hidden_next_cnt

        weights2 = tf.Variable(tf.truncated_normal([hidden_cur_cnt, num_labels], stddev=hidden_stddev / 2))
        biases2 = tf.Variable(tf.zeros([num_labels]))

        # Training computation. right most
        y0 = tf.matmul(tf_train_dataset, weights1) + biases1
        hidden = tf.nn.relu(y0)
        hidden_drop = hidden

        valid_y0 = tf.matmul(tf_valid_dataset, weights1) + biases1
        valid_hidden = tf.nn.relu(valid_y0)

        test_y0 = tf.matmul(tf_test_dataset, weights1) + biases1
        test_hidden = tf.nn.relu(test_y0)

        # enable DropOut
        keep_prob = 0.5
        if drop_out:
            hidden_drop = tf.nn.dropout(hidden, keep_prob)
            hidden_drop = hidden_drop
        y0s = []
        y0s.append(y0)
        y1s = []

        hiddens = []
        hiddens.append(hidden)
        hidden_drops = []
        hidden_drops.append(hidden_drop)
        valid_y0s = []
        valid_y0s.append(valid_y0)
        valid_hiddens = []
        valid_hiddens.append(valid_hidden)
        test_y0s = []
        test_y0s.append(test_y0)
        test_hiddens = []
        test_hiddens.append(test_hidden)

        # middle layer
        for i in range(layer_cnt - 2):
            y1s.append(tf.matmul(hidden_drops[i], weights[i]) + biases[i])
            hidden_drops.append(tf.nn.relu(y1s[i]))
            if drop_out:
                keep_prob += 0.5 * (i + 1) / float(layer_cnt)
                hidden_drops[i + 1] = tf.nn.dropout(hidden_drops[i + 1], keep_prob)

            y0s.append(tf.matmul(hiddens[i], weights[i]) + biases[i])
            hiddens.append(tf.nn.relu(y0s[i + 1]))

            valid_y0s.append(tf.matmul(valid_hiddens[i], weights[i]) + biases[i])
            valid_hiddens.append(tf.nn.relu(valid_y0s[i + 1]))

            test_y0s.append(tf.matmul(test_hiddens[i], weights[i]) + biases[i])
            test_hiddens.append(tf.nn.relu(test_y0s[i + 1]))

        logits = tf.matmul(hidden_drops[layer_cnt - 2], weights2) + biases2
        # only drop out when train
        logits_predict = tf.matmul(hiddens[layer_cnt - 2], weights2) + biases2
        valid_predict = tf.matmul(valid_hiddens[layer_cnt - 2], weights2) + biases2
        test_predict = tf.matmul(test_hiddens[layer_cnt - 2], weights2) + biases2

        l2_loss = 0
        # enable regularization
        if regular:
            l2_loss = tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2)
            for i in range(len(weights)):
                l2_loss += tf.nn.l2_loss(weights[i])
                # l2_loss += tf.nn.l2_loss(biases[i])
            beta = 1.0 / image_size / image_size
            l2_loss *= beta
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) + l2_loss
        loss *= 10
        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
        if lrd:
            cur_step = tf.Variable(0)  # count the number of steps taken.
            starter_learning_rate = 0.1
            learning_rate = tf.train.exponential_decay(starter_learning_rate, cur_step, 100000, 0.96, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=cur_step)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits_predict)
        valid_prediction = tf.nn.softmax(valid_predict)
        test_prediction = tf.nn.softmax(test_predict)

    num_steps = 40001

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print("Initialized")
        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            if offset_range == -1:
                offset_range = train_labels.shape[0] - batch_size

            offset = (step * batch_size) % offset_range
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if step % 100 == 0:
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(
                    valid_prediction.eval(), valid_labels))
        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


if __name__ == '__main__':
    image_size = 28
    num_labels = 10
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = \
        load_reformat_not_mnist(image_size, num_labels)
    # tf_better_nn(regular=True)
    # tf_better_nn(offset_range=1000)
    # tf_better_nn(offset_range=1000, drop_out=True)
    # tf_better_nn(lrd=True)
    tf_deep_nn(layer_cnt=4, lrd=False, drop_out=False, regular=False)
    # tf_deep_nn(layer_cnt=4)
