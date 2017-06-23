from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from app.caltech.data import read_caltech
from util.board import variable_summary


def recall_rate(predictions, labels, is_test=False):
    threshold = 0.0
    i_predictions = np.subtract(predictions, np.asarray([[threshold, 0] for _ in range(len(labels))]))
    predict_succ = np.argmax(i_predictions, 1) == np.argmax(labels, 1)
    pos_samples = np.argmax(labels, 1) == 0
    res = 100.0 * np.sum(np.logical_and(predict_succ, pos_samples)) / np.sum(pos_samples)
    if is_test:
        for i in range(20):
            threshold = -1.0 + i * 0.1
            i_predictions = np.subtract(predictions, np.asarray([[threshold, 0] for _ in range(len(labels))]))
            predict_succ = np.argmax(i_predictions, 1) == np.argmax(labels, 1)
            pos_samples = np.argmax(labels, 1) == 0
            res = 100.0 * np.sum(np.logical_and(predict_succ, pos_samples)) / np.sum(pos_samples)
            print('recall: %f%%' % res)
    return res


def accuracy(predictions, labels, is_test=False):
    if is_test:
        for i in range(20):
            threshold = -1.0 + i * 0.1
            i_predictions = np.subtract(predictions, np.asarray([[threshold, 0] for _ in range(len(labels))]))
            res = 100.0 * np.sum(np.argmax(i_predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]
            print('precision: %f%%' % res)
    else:
        threshold = 0.0
        i_predictions = np.subtract(predictions, np.asarray([[threshold, 0] for _ in range(len(labels))]))
        res = 100.0 * np.sum(np.argmax(i_predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]
    return res


def tf_deep_nn(regular=False, test=False):
    batch_size = 128

    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope('input'):
            with tf.name_scope('feature'):
                tf_train_dataset = tf.placeholder(tf.float32, shape=(None, feature_dim))
                variable_summary(tf_train_dataset)
            with tf.name_scope('label'):
                tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels))
                variable_summary(tf_train_labels)

        hidden_node_count = 32
        # start weight
        hidden_stddev = np.sqrt(2.0 / 100)
        with tf.name_scope('hidden1'):
            with tf.name_scope('weight'):
                weights1 = tf.Variable(tf.truncated_normal([feature_dim, hidden_node_count], stddev=hidden_stddev))
                # variable_summary(weights1)
            with tf.name_scope('biases'):
                biases1 = tf.Variable(tf.zeros([hidden_node_count]))
                # variable_summary(biases1)
            # first wx + b
            with tf.name_scope('wx_b'):
                y0 = tf.matmul(tf_train_dataset, weights1) + biases1
                # variable_summary(y0)
            # first sigmoid
            with tf.name_scope('sigmoid'):
                hidden = tf.nn.sigmoid(y0)
                # variable_summary(hidden)

        with tf.name_scope('hidden2'):
            hidden_cur_cnt = hidden_node_count
            hidden_next_cnt = int(hidden_cur_cnt / 2)
            with tf.name_scope('weight'):
                weights2 = tf.Variable(tf.truncated_normal([hidden_cur_cnt, hidden_next_cnt], stddev=hidden_stddev))
                # variable_summary(weights2)
            with tf.name_scope('biases'):
                biases2 = tf.Variable(tf.zeros([hidden_next_cnt]))
                # variable_summary(biases2)
            # first wx + b
            with tf.name_scope('wx_b'):
                y0 = tf.matmul(hidden, weights2) + biases2
                # variable_summary(y0)
            # first sigmoid
            with tf.name_scope('sigmoid'):
                hidden = tf.nn.sigmoid(y0)
                # variable_summary(hidden)

        # last weight
        with tf.name_scope('hidden_3'):
            with tf.name_scope('weight'):
                weights3 = tf.Variable(tf.truncated_normal([hidden_next_cnt, num_labels], stddev=hidden_stddev / 2))
                # variable_summary(weights3)
            with tf.name_scope('biases'):
                biases3 = tf.Variable(tf.zeros([num_labels]))
                # variable_summary(biases3)
            # last wx + b
            with tf.name_scope('wx_b'):
                logits = tf.matmul(hidden, weights3) + biases3
                # variable_summary(biases3)
            with tf.name_scope('sigmoid'):
                logits = tf.nn.sigmoid(logits)
                # variable_summary(logits)

        logits_predict = logits

        l2_loss = 0
        # enable regularization
        with tf.name_scope('loss'):
            if regular:
                with tf.name_scope('l2_norm'):
                    l2_loss = tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2) + tf.nn.l2_loss(weights3)
                    beta = 1e-2
                    l2_loss *= beta

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
            loss += l2_loss
            with tf.name_scope('summaries'):
                tf.summary.histogram('histogram', loss)
        # Optimizer.
        with tf.name_scope('Gradient'):
            cur_step = tf.Variable(0, trainable=False)  # count the number of steps taken.
            starter_learning_rate = 0.4
            learning_rate = tf.train.exponential_decay(starter_learning_rate, cur_step, 500, 0.75, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=cur_step)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits_predict)

        saver = tf.train.Saver()

        merged = tf.summary.merge_all()
    summary_flag = True
    summary_dir = 'summary'
    if tf.gfile.Exists(summary_dir):
        tf.gfile.DeleteRecursively(summary_dir)
    tf.gfile.MakeDirs(summary_dir)
    num_steps = 5001

    save_path = 'ct_save.ckpt'
    save_flag = True

    with tf.Session(graph=graph) as session:
        train_writer = tf.summary.FileWriter(summary_dir + '/train',
                                             session.graph)
        test_writer = tf.summary.FileWriter(summary_dir + '/test')
        if os.path.exists(save_path + '.index') and save_flag:
            # Restore variables from disk.
            saver.restore(session, './' + save_path)
            print('restore')
        else:
            tf.global_variables_initializer().run()
            print("Initialized")
        if not test:
            for step in range(num_steps):
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
                    print("Minibatch accuracy: %.1f%%, recall: %.1f%%" % (
                        accuracy(predictions, batch_labels), recall_rate(predictions, batch_labels))
                          )
                    print("Validate accuracy: %.1f%%, recall: %.1f%%" % (accuracy(
                        train_prediction.eval(
                            feed_dict={tf_train_dataset: valid_dataset, tf_train_labels: valid_labels}),
                        valid_labels), recall_rate(
                        train_prediction.eval(
                            feed_dict={tf_train_dataset: valid_dataset,
                                       tf_train_labels: valid_labels}),
                        valid_labels)))
                    if summary_flag:
                        test_writer.add_summary(summary, step)
                if summary_flag:
                    train_writer.add_summary(summary, step)
        print("Test accuracy: %.1f%%, recall: %.1f%%" % (accuracy(
            train_prediction.eval(
                feed_dict={tf_train_dataset: test_dataset, tf_train_labels: test_labels}),
            test_labels, is_test=True), recall_rate(
            train_prediction.eval(
                feed_dict={tf_train_dataset: test_dataset,
                           tf_train_labels: test_labels}),
            test_labels, is_test=True)))

        if save_flag and not test:
            saver.save(session, save_path)


if __name__ == '__main__':
    feature_dim = 2330
    num_labels = 2
    raw_dataset, raw_labels, test_dataset, test_labels = read_caltech()
    raw_train_size = len(raw_dataset)

    train_dataset = raw_dataset[: raw_train_size / 2]
    train_labels = raw_labels[: raw_train_size / 2]

    valid_dataset = raw_dataset[raw_train_size / 2:]
    valid_labels = raw_labels[raw_train_size / 2:]
    train_dataset = np.asarray(train_dataset, dtype=np.float32)
    train_labels = np.asarray(train_labels, dtype=np.int32)
    valid_dataset = np.asarray(valid_dataset, dtype=np.float32)
    valid_labels = np.asarray(valid_labels, dtype=np.int32)
    test_dataset = np.asarray(test_dataset, dtype=np.float32)
    test_labels = np.asarray(test_labels, dtype=np.int32)
    tf_deep_nn(regular=True, test=True)

# tensorboard --logdir=src/app/caltech/summary
