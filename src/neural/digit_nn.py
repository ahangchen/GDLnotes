import random
import numpy as np
import tensorflow as tf


def div(xt):
    label1 = int(abs(xt[0]) < 0.5)
    label2 = int(abs(xt[1]) < 0.5)
    return label1 + label2


def train_data():
    inputs = [[random.uniform(-1, 1), random.uniform(-1, 1)] for i in range(100000)]
    labels = np.asarray([div(x_t) for x_t in inputs])
    labels = (np.arange(3) == labels[:, None]).astype(np.float32)

    print(inputs[0])
    print(div(inputs[0]))
    print(labels[0])
    return inputs, labels


def accuracy(predictions, train_labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(train_labels, 1)) / predictions.shape[0]


def dig_nn(dataset, train_labels, batch_size, data_count, label_count):
    graph = tf.Graph()
    with graph.as_default():
        tf_train_dataset = tf.placeholder(tf.float32,
                                          shape=(batch_size, data_count))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, label_count))
        hidden_node_count = [10, 10]
        wi = tf.Variable(tf.truncated_normal([data_count, hidden_node_count[0]]))
        bi = tf.Variable(tf.zeros([hidden_node_count[0]]))

        y1 = tf.matmul(tf_train_dataset, wi) + bi
        h1 = tf.nn.relu(y1)

        w0 = tf.Variable(tf.truncated_normal([hidden_node_count[0], hidden_node_count[1]]))
        b0 = tf.Variable(tf.zeros([hidden_node_count[1]]))

        y2 = tf.matmul(h1, w0) + b0
        h2 = tf.nn.relu(y2)

        wo = tf.Variable(tf.truncated_normal([hidden_node_count[1], label_count]))
        bo = tf.Variable(tf.zeros([label_count]))

        logits = tf.matmul(h2, wo) + bo
        train_prediction = tf.nn.softmax(logits)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    num_steps = 1000

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print("Initialized")
        for step in range(num_steps):
            batch_data = dataset[step * batch_size: (step + 1) * batch_size]
            batch_labels = train_labels[step * batch_size: (step + 1) * batch_size]

            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if step % 10 == 0:
                print('=' * 80)
                cur_first_data = dataset[step * batch_size: (step + 1) * batch_size][0]
                print('current first data [%f, %f]' % (cur_first_data[0], cur_first_data[1]))
                print('current first predict: [%f, %f, %f]' % (predictions[0][0], predictions[0][1], predictions[0][2]))
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))

if __name__ == '__main__':
    inputs, labels = train_data()
    dig_nn(inputs, labels, 100, 2, 3)
