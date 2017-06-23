from __future__ import print_function


import tensorflow as tf

from convnet.conv_mnist import maxpool2d
from util.board import variable_summary
from util.mnist import format_mnist


def large_data_size(data):
    return data.get_shape()[1] > 1 and data.get_shape()[2] > 1


def conv_train(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels, image_size,
               num_labels, basic_hps, stride_ps, drop=False, lrd=False, get_grad=False, norm_list=None):
    batch_size = basic_hps['batch_size']
    patch_size = basic_hps['patch_size']
    depth = basic_hps['depth']
    first_hidden_num = 192
    second_hidden_num = basic_hps['num_hidden']
    num_channels = 1
    layer_cnt = basic_hps['layer_sum']
    loss_collect = list()

    graph = tf.Graph()
    with graph.as_default():
        # Input data.
        with tf.name_scope('input'):
            with tf.name_scope('data'):
                tf_train_dataset = tf.placeholder(
                    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
                variable_summary(tf_train_dataset)
            with tf.name_scope('label'):
                tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
                variable_summary(tf_train_labels)

        # Variables.
        # the third parameter must be same as the last layer depth
        with tf.name_scope('input_cnn_filter'):
            with tf.name_scope('input_weight'):
                input_weights = tf.Variable(tf.truncated_normal(
                    [patch_size, patch_size, num_channels, depth], stddev=0.1), name='input_weight')
                variable_summary(input_weights)
            with tf.name_scope('input_biases'):
                input_biases = tf.Variable(tf.zeros([depth]), name='input_biases')
                variable_summary(input_weights)

        mid_layer_cnt = layer_cnt - 1
        layer_weights = list()
        layer_biases = [tf.Variable(tf.constant(1.0, shape=[depth * (i + 2)])) for i in range(mid_layer_cnt)]
        for i in range(mid_layer_cnt):
            variable_summary(layer_biases)
        output_weights = list()
        output_biases = tf.Variable(tf.constant(1.0, shape=[first_hidden_num]))
        with tf.name_scope('first_nn'):
            with tf.name_scope('weights'):
                first_nn_weights = tf.Variable(tf.truncated_normal(
                    [first_hidden_num, second_hidden_num], stddev=0.1))
                variable_summary(first_nn_weights)
            with tf.name_scope('biases'):
                first_nn_biases = tf.Variable(tf.constant(1.0, shape=[second_hidden_num]))
                variable_summary(first_nn_weights)
        with tf.name_scope('second_nn'):
            with tf.name_scope('weights'):
                second_nn_weights = tf.Variable(tf.truncated_normal(
                    [second_hidden_num, num_labels], stddev=0.1))
                variable_summary(second_nn_weights)
            with tf.name_scope('biases'):
                second_nn_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
                variable_summary(second_nn_biases)

        # Model.
        def model(data, init=True):
            if not large_data_size(data) or not large_data_size(input_weights):
                stride_ps[0] = [1, 1, 1, 1]
            with tf.name_scope('first_cnn'):
                conv = tf.nn.conv2d(data, input_weights, stride_ps[0], use_cudnn_on_gpu=True, padding='SAME')
                if init:
                    print('init')
                    variable_summary(conv)
            with tf.name_scope('first_max_pool'):
                conv = maxpool2d(conv)
                if init:
                    variable_summary(conv)
            hidden = tf.nn.relu6(conv + input_biases)
            if init:
                tf.summary.histogram('first_act', hidden)
            if drop and init:
                with tf.name_scope('first_drop'):
                    hidden = tf.nn.dropout(hidden, 0.8, name='drop1')
                    tf.summary.histogram('first_drop', hidden)
            for i in range(mid_layer_cnt):
                with tf.name_scope('cnn{i}'.format(i=i)):
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
                        with tf.name_scope('weight'.format(i=i)):
                            layer_weight = tf.Variable(tf.truncated_normal(
                                shape=[filter_w, filter_h, depth * (i + 1), depth * (i + 2)], stddev=0.1))
                            variable_summary(layer_weight)
                        layer_weights.append(layer_weight)
                    if not large_data_size(hidden) or not large_data_size(layer_weights[i]):
                        # print("is not large data")
                        stride_ps[i + 1] = [1, 1, 1, 1]
                    # print(stride_ps[i + 1])
                    # print(len(stride_ps))
                    # print(i + 1)
                    with tf.name_scope('conv2d'):
                        conv = tf.nn.conv2d(hidden, layer_weights[i], stride_ps[i + 1], use_cudnn_on_gpu=True, padding='SAME')
                        if init:
                            variable_summary(conv)
                    with tf.name_scope('maxpool2d'):
                        if not large_data_size(conv):
                            print('not large')
                            conv = maxpool2d(conv, 1, 1)
                            if init:
                                variable_summary(conv)
                        else:
                            conv = maxpool2d(conv)
                            if init:
                                variable_summary(conv)
                    with tf.name_scope('act'):
                        hidden = tf.nn.relu6(conv + layer_biases[i])
                        if init:
                            variable_summary(conv)

            shapes = hidden.get_shape().as_list()
            shape_mul = 1
            for s in shapes[1:]:
                shape_mul *= s

            if init:
                with tf.name_scope('output'):
                    output_size = shape_mul
                    with tf.name_scope('weights'):
                        output_weights.append(tf.Variable(tf.truncated_normal([output_size, first_hidden_num], stddev=0.1)))
                        variable_summary(output_weights)
            reshape = tf.reshape(hidden, [shapes[0], shape_mul])
            with tf.name_scope('output_act'):
                hidden = tf.nn.relu6(tf.matmul(reshape, output_weights[0]) + output_biases)
                if init:
                    tf.summary.histogram('output_act', hidden)
            if drop and init:
                with tf.name_scope('output_drop'):
                    hidden = tf.nn.dropout(hidden, 0.5)
                    tf.summary.histogram('output_drop', hidden)
            with tf.name_scope('output_wx_b'):
                hidden = tf.matmul(hidden, first_nn_weights) + first_nn_biases
                if init:
                    tf.summary.histogram('output_wx_b', hidden)
            if drop and init:
                with tf.name_scope('final_drop'):
                    hidden = tf.nn.dropout(hidden, 0.5)
                    tf.summary.histogram('final_drop', hidden)
            with tf.name_scope('final_wx_b'):
                hidden = tf.matmul(hidden, second_nn_weights) + second_nn_biases
                if init:
                    tf.summary.histogram('final_wx_b', hidden)
            return hidden

        # Training computation.
        with tf.name_scope('logits'):
            logits = model(tf_train_dataset)
            tf.summary.histogram('logits', logits)
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
            tf.summary.histogram('loss', loss)
        # Optimizer.

        with tf.name_scope('train'):
            if lrd:
                cur_step = tf.Variable(0)  # count the number of steps taken.
                starter_learning_rate = 0.06
                learning_rate = tf.train.exponential_decay(starter_learning_rate, cur_step, 600, 0.1, staircase=True)
                optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=cur_step)
            else:
                optimizer = tf.train.AdagradOptimizer(0.1).minimize(loss)

        # Predictions for the training, validation, and test data.
        with tf.name_scope('train_predict'):
            train_prediction = tf.nn.softmax(logits)
            variable_summary(train_prediction)
        # with tf.name_scope('valid_predict'):
        #     valid_prediction = tf.nn.softmax(model(tf_valid_dataset, init=False))
        #     variable_summary(valid_prediction, 'valid_predict')
        # with tf.name_scope('test_predict'):
        #     test_prediction = tf.nn.softmax(model(tf_test_dataset, init=False))
        #     variable_summary(test_prediction, 'test_predict')
        merged = tf.summary.merge_all()
    summary_flag = True
    summary_dir = 'summary'
    if tf.gfile.Exists(summary_dir):
        tf.gfile.DeleteRecursively(summary_dir)
    tf.gfile.MakeDirs(summary_dir)

    num_steps = 5001
    with tf.Session(graph=graph) as session:
        train_writer = tf.summary.FileWriter(summary_dir + '/train',
                                             session.graph)
        valid_writer = tf.summary.FileWriter(summary_dir + '/valid')
        tf.global_variables_initializer().run()
        print('Initialized')
        mean_loss = 0
        for step in range(num_steps):
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            if summary_flag:
                summary, _, l, predictions = session.run(
                    [merged, optimizer, loss, train_prediction], options=run_options, feed_dict=feed_dict)
            else:
                _, l, predictions = session.run(
                    [optimizer, loss, train_prediction], options=run_options, feed_dict=feed_dict)
            mean_loss += l
            if step % 5 == 0:
                mean_loss /= 5.0
                loss_collect.append(mean_loss)
                mean_loss = 0
                if step % 50 == 0:
                    print('Minibatch loss at step %d: %f' % (step, l))
                    # print('Validation accuracy: %.1f%%' % accuracy(
                    #     valid_prediction.eval(), valid_labels))
                    if step % 100 == 0 and summary_flag:
                        train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
                        train_writer.add_summary(summary, step)
                        print('Adding run metadata for', step)
                if summary_flag:
                    valid_writer.add_summary(summary, step)
            if summary_flag:
                train_writer.add_summary(summary, step)
        train_writer.close()
        valid_writer.close()
        # print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


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
