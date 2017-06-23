from __future__ import print_function

import os

import tensorflow as tf

from convnet.conv_mnist import maxpool2d
from neural.full_connect import accuracy
from util.mnist import format_mnist


def large_data_size(data):
    return data.get_shape()[1] > 1 and data.get_shape()[2] > 1


def conv_train(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels, image_size,
               num_labels, basic_hps, stride_ps, drop=False, lrd=False):
    batch_size = basic_hps['batch_size']
    patch_size = basic_hps['patch_size']
    depth = basic_hps['depth']
    first_hidden_num = basic_hps['num_hidden']
    second_hidden_num = first_hidden_num / 2 + 1
    num_channels = 1
    layer_cnt = basic_hps['layer_sum']

    graph = tf.Graph()
    with graph.as_default():
        # Input data.
        tf_train_dataset = tf.placeholder(
            tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Variables.
        # the third parameter must be same as the last layer depth
        input_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, num_channels, depth], stddev=0.1))
        input_biases = tf.Variable(tf.zeros([depth]))

        mid_layer_cnt = layer_cnt - 1
        layer_weights = list()
        layer_biases = [tf.Variable(tf.constant(1.0, shape=[depth * (i + 2)])) for i in range(mid_layer_cnt)]
        output_weights = list()
        output_biases = tf.Variable(tf.constant(1.0, shape=[first_hidden_num]))
        first_nn_weights = tf.Variable(tf.truncated_normal(
            [first_hidden_num, second_hidden_num], stddev=0.1))
        second_nn_weights = tf.Variable(tf.truncated_normal(
            [second_hidden_num, num_labels], stddev=0.1))
        first_nn_biases = tf.Variable(tf.constant(1.0, shape=[second_hidden_num]))
        second_nn_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

        # Model.
        def model(data, model_drop=True, init=True):
            if not large_data_size(data) or not large_data_size(input_weights):
                stride_ps[0] = [1, 1, 1, 1]
            conv = tf.nn.conv2d(data, input_weights, stride_ps[0], use_cudnn_on_gpu=True, padding='SAME')
            conv = maxpool2d(conv)
            hidden = tf.nn.relu6(conv + input_biases)
            if drop and model_drop:
                hidden = tf.nn.dropout(hidden, 0.8)
            for i in range(mid_layer_cnt):
                print(hidden)
                if init:
                    # avoid filter shape larger than input shape
                    hid_shape = hidden.get_shape()
                    # print(hid_shape)
                    filter_w = patch_size / (i + 1)
                    filter_h = patch_size / (i + 1)
                    # print(filter_w)
                    # print(filter_h)
                    if filter_w > hid_shape[1]:
                        filter_w = int(hid_shape[1])
                    if filter_h > hid_shape[2]:
                        filter_h = int(hid_shape[2])
                    layer_weight = tf.Variable(tf.truncated_normal(
                        shape=[filter_w, filter_h, depth * (i + 1), depth * (i + 2)], stddev=0.1))
                    layer_weights.append(layer_weight)
                if not large_data_size(hidden) or not large_data_size(layer_weights[i]):
                    # print("is not large data")
                    stride_ps[i + 1] = [1, 1, 1, 1]
                # print(stride_ps[i + 1])
                # print(len(stride_ps))
                # print(i + 1)
                conv = tf.nn.conv2d(hidden, layer_weights[i], stride_ps[i + 1], use_cudnn_on_gpu=True, padding='SAME')
                if not large_data_size(conv):
                    print('not large')
                    conv = maxpool2d(conv, 1, 1)
                else:
                    conv = maxpool2d(conv)
                hidden = tf.nn.relu6(conv + layer_biases[i])

            shapes = hidden.get_shape().as_list()
            shape_mul = 1
            for s in shapes[1:]:
                shape_mul *= s

            if init:
                output_size = shape_mul
                output_weights.append(tf.Variable(tf.truncated_normal([output_size, first_hidden_num], stddev=0.1)))
            reshape = tf.reshape(hidden, [shapes[0], shape_mul])

            hidden = tf.nn.relu6(tf.matmul(reshape, output_weights[0]) + output_biases)
            if drop and model_drop:
                hidden = tf.nn.dropout(hidden, 0.5)
            hidden = tf.matmul(hidden, first_nn_weights) + first_nn_biases
            if drop and model_drop:
                hidden = tf.nn.dropout(hidden, 0.5)
            hidden = tf.matmul(hidden, second_nn_weights) + second_nn_biases
            return hidden

        # Training computation.
        logits = model(tf_train_dataset)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
        # Optimizer.
        if lrd:
            cur_step = tf.Variable(0)  # count the number of steps taken.
            starter_learning_rate = 0.1
            learning_rate = tf.train.exponential_decay(starter_learning_rate, cur_step, 600, 0.1, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=cur_step)
        else:
            optimizer = tf.train.AdagradOptimizer(0.06).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(model(tf_valid_dataset, model_drop=False, init=False))
        test_prediction = tf.nn.softmax(model(tf_test_dataset, model_drop=False, init=False))
        saver = tf.train.Saver()
    # on step 1750, run over 55000 train images
    num_steps = 1750 * 3

    save_path = 'conv_mnist'
    save_flag = True
    with tf.Session(graph=graph) as session:
        if os.path.exists(save_path) and save_flag:
            # Restore variables from disk.
            saver.restore(session, save_path)
        else:
            tf.global_variables_initializer().run()
            print('Initialized')
        end_train = False
        mean_loss = 0
        for step in range(num_steps):
            if end_train:
                break
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
            mean_loss += l
            if step % 10 == 0:
                mean_loss /= 10.0
                if step % 200 == 0:
                    print('Minibatch loss at step %d: %f' % (step, mean_loss))
                    print('Validation accuracy: %.1f%%' % accuracy(
                        valid_prediction.eval(), valid_labels))
                mean_loss = 0
        if save_flag:
            saver.save(session, save_path)
        print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


def hp_train():
    image_size = 28
    num_labels = 10
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = \
        format_mnist()
    pick_size = 2048
    valid_dataset = valid_dataset[0: pick_size, :, :, :]
    valid_labels = valid_labels[0: pick_size, :]
    test_dataset = test_dataset[0: pick_size, :, :, :]
    test_labels = test_labels[0: pick_size, :]
    basic_hypers = {
        'batch_size': 32,
        'patch_size': 5,
        'depth': 16,
        'num_hidden': 64,
        'layer_sum': 2
    }
    stride_params = [[1, 2, 2, 1] for _ in range(basic_hypers['layer_sum'])]
    conv_train(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset,
               test_labels,
               image_size, num_labels, basic_hypers, stride_params, drop=True, lrd=False)


if __name__ == '__main__':
    hp_train()
