import tensorflow as tf

from convnet.conv_mnist import maxpool2d, load_reformat_not_mnist
from neural.full_connect import accuracy


def conv_train(basic_hps, stride_p, drop=False, lrd=False):
    batch_size = basic_hps['batch_size']
    patch_size = basic_hps['patch_size']
    depth = basic_hps['depth']
    num_hidden = basic_hps['num_hidden']
    num_channels = basic_hps['num_channels']

    graph = tf.Graph()
    with graph.as_default():
        # Input data.
        tf_train_dataset = tf.placeholder(
            tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Variables.
        input_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, num_channels, depth], stddev=0.1))
        input_biases = tf.Variable(tf.zeros([depth]))

        layer2_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, depth, depth], stddev=0.1))
        layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

        output_weights = tf.Variable(tf.truncated_normal(
            [64, num_hidden], stddev=0.1))
        output_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
        final_weights = tf.Variable(tf.truncated_normal(
            [num_hidden, num_labels], stddev=0.1))
        final_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

        # Model.
        def model(data):
            conv = tf.nn.conv2d(data, input_weights, stride_p, padding='SAME')
            conv = maxpool2d(conv)
            hidden = tf.nn.relu(conv + input_biases)
            if drop:
                hidden = tf.nn.dropout(hidden, 0.5)

            conv = tf.nn.conv2d(hidden, layer2_weights, stride_p, padding='SAME')
            conv = maxpool2d(conv)
            hidden = tf.nn.relu(conv + layer2_biases)
            if drop:
                hidden = tf.nn.dropout(hidden, 0.7)

            shape = hidden.get_shape().as_list()
            reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
            hidden = tf.nn.relu(tf.matmul(reshape, output_weights) + output_biases)
            if drop:
                hidden = tf.nn.dropout(hidden, 0.8)
            return tf.matmul(hidden, final_weights) + final_biases

        # Training computation.
        logits = model(tf_train_dataset)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
        # Optimizer.
        if lrd:
            cur_step = tf.Variable(0)  # count the number of steps taken.
            starter_learning_rate = 0.1
            learning_rate = tf.train.exponential_decay(starter_learning_rate, cur_step, 10000, 0.96, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=cur_step)
        else:
            optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
        test_prediction = tf.nn.softmax(model(tf_test_dataset))
    num_steps = 5001

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if step % 50 == 0:
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                print('Validation accuracy: %.1f%%' % accuracy(
                    valid_prediction.eval(), valid_labels))
        print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


if __name__ == '__main__':
    image_size = 28
    num_labels = 10
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = \
        load_reformat_not_mnist(image_size, num_labels, 1)
    # conv_max_pool_train()
    # conv_train()
    basic_hypers = {
        'batch_size': 16,
        'patch_size': 5,
        'depth': 16,
        'num_hidden': 64,
        'num_channels': 1,
    }
    stride_param = [1, 3, 2, 1]
    conv_train(basic_hypers, stride_param, lrd=True)
