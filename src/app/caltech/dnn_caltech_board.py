from __future__ import print_function

import numpy as np
import tensorflow as tf

from app.caltech.data import read_caltech
from neural.full_connect import accuracy
from util.board import variable_summary


def tf_deep_nn(regular=False, drop_out=False, lrd=False, layer_cnt=2):
    batch_size = 128

    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope('input'):
            with tf.name_scope('feature'):
                tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, feature_dim))
                variable_summary(tf_train_dataset)
            with tf.name_scope('label'):
                tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
                variable_summary(tf_train_labels)
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        hidden_node_count = 32
        # start weight
        hidden_stddev = np.sqrt(2.0 / 100)
        with tf.name_scope('hidden1'):
            with tf.name_scope('weight'):
                weights1 = tf.Variable(tf.truncated_normal([feature_dim, hidden_node_count], stddev=hidden_stddev))
                variable_summary(weights1)
            with tf.name_scope('biases'):
                biases1 = tf.Variable(tf.zeros([hidden_node_count]))
                variable_summary(biases1)
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
            with tf.name_scope('hidden%d' % (i + 2)):
                with tf.name_scope('weight'):
                    w_i = tf.Variable(tf.truncated_normal([hidden_cur_cnt, hidden_next_cnt], stddev=hidden_stddev))
                    weights.append(w_i)
                    variable_summary(w_i)
                with tf.name_scope('biases'):
                    b_i = tf.Variable(tf.zeros([hidden_next_cnt]))
                    biases.append(b_i)
                    variable_summary(b_i)
            hidden_cur_cnt = hidden_next_cnt
        # first wx + b
        with tf.name_scope('hidden1/wx_b'):
            y0 = tf.matmul(tf_train_dataset, weights1) + biases1
            variable_summary(y0)
        # first sigmoid
        with tf.name_scope('hidden1/sigmoid'):
            hidden = tf.nn.sigmoid(y0)
            variable_summary(hidden)
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
            with tf.name_scope('hidden%d' % (i + 2)):
                y0 = tf.matmul(hidden, weights[i]) + biases[i]
                variable_summary(y0)
            with tf.name_scope('hidden%d' % (i + 2)):
                hidden = tf.nn.sigmoid(y0)
                variable_summary(hidden)

            # hidden = y0

            valid_y0 = tf.matmul(valid_hidden, weights[i]) + biases[i]
            valid_hidden = tf.nn.sigmoid(valid_y0)
            # valid_hidden = valid_y0

            test_y0 = tf.matmul(test_hidden, weights[i]) + biases[i]
            test_hidden = tf.nn.sigmoid(test_y0)
            # test_hidden = test_y0

        # last weight
        with tf.name_scope('one_hot'):
            with tf.name_scope('weight'):
                weights2 = tf.Variable(tf.truncated_normal([hidden_cur_cnt, num_labels], stddev=hidden_stddev / 2))
                variable_summary(weights2)
            with tf.name_scope('biases'):
                biases2 = tf.Variable(tf.zeros([num_labels]))
                variable_summary(biases2)
            # last wx + b
            with tf.name_scope('output'):
                logits = tf.matmul(hidden_drop, weights2) + biases2
                variable_summary(biases2)

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
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) + l2_loss
            tf.summary.histogram('loss', loss)
        # Optimizer.
        with tf.name_scope('train'):
            if lrd:
                cur_step = tf.Variable(0, trainable=False)  # count the number of steps taken.
                starter_learning_rate = 0.4
                learning_rate = tf.train.exponential_decay(starter_learning_rate, cur_step, 500, 0.75, staircase=True)
                optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=cur_step)
            else:
                optimizer = tf.train.AdamOptimizer(0.5).minimize(loss)

        # Predictions for the training, validation, and test data.
        with tf.name_scope('train_predict'):
            train_prediction = tf.nn.softmax(logits_predict)
            variable_summary(train_prediction)
        valid_prediction = tf.nn.softmax(valid_predict)
        test_prediction = tf.nn.softmax(test_predict)

        merged = tf.summary.merge_all()
    summary_flag = True
    summary_dir = 'summary'
    if tf.gfile.Exists(summary_dir):
        tf.gfile.DeleteRecursively(summary_dir)
    tf.gfile.MakeDirs(summary_dir)
    num_steps = 8001

    with tf.Session(graph=graph) as session:
        train_writer = tf.summary.FileWriter(summary_dir + '/train',
                                             session.graph)
        tf.global_variables_initializer().run()
        print("Initialized")
        for step in range(num_steps):
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            offset_range = train_labels.shape[0] - batch_size
            offset = (step * batch_size) % offset_range
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            if summary_flag:
                summary, _, l, predictions = session.run(
                    [merged, optimizer, loss, train_prediction], feed_dict=feed_dict)
            else:
                _, l, predictions = session.run(
                    [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if step % 50 == 0:
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(
                    valid_prediction.eval(), valid_labels))
                if summary_flag:
                    train_writer.add_summary(summary, step)
                    print('Adding run metadata for', step)
            if summary_flag:
                train_writer.add_summary(summary, step)
            train_writer.close()
        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


if __name__ == '__main__':
    feature_dim = 2330
    num_labels = 2
    raw_dataset, raw_labels, test_dataset, test_labels = read_caltech()
    raw_train_size = len(raw_dataset)

    train_dataset = raw_dataset[: raw_train_size/2]
    train_labels = raw_labels[: raw_train_size/2]

    valid_dataset = raw_dataset[raw_train_size/2:]
    valid_labels = raw_labels[raw_train_size/2:]
    train_dataset = np.asarray(train_dataset, dtype=np.float32)
    train_labels = np.asarray(train_labels, dtype=np.int32)
    valid_dataset = np.asarray(valid_dataset, dtype=np.float32)
    valid_labels = np.asarray(valid_labels, dtype=np.int32)
    test_dataset = np.asarray(test_dataset, dtype=np.float32)
    test_labels = np.asarray(test_labels, dtype=np.int32)
    tf_deep_nn(layer_cnt=3, lrd=True, drop_out=False, regular=True)

# tensorboard --logdir=src/app/caltech/summary
