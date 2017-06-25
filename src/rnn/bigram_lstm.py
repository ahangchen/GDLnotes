# coding=utf-8
import random
import string
import zipfile

import numpy as np
import tensorflow as tf

from not_mnist.img_pickle import save_obj, load_pickle
from not_mnist.load_data import maybe_download


def read_data(filename):
    f = zipfile.ZipFile(filename)
    for name in f.namelist():
        return tf.compat.as_str(f.read(name))
    f.close()


data_set = load_pickle('text8_text.pickle')
if data_set is None:
    # load data
    url = 'http://mattmahoney.net/dc/'
    filename = maybe_download('text8.zip', 31344016, url=url)

    # read data
    text = read_data(filename)
    print('Data size %d' % len(text))
    save_obj('text8_text.pickle', text)
else:
    text = data_set

# Create a small validation set.
valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
print(train_size, train_text[:64])
print(valid_size, valid_text[:64])

vocabulary_size = (len(string.ascii_lowercase) + 1) * (len(string.ascii_lowercase) + 1)  # [a-z] + ' '

idx2bi = {}
bi2idx = {}
idx = 0
for i in ' ' + string.ascii_lowercase:
    for j in ' ' + string.ascii_lowercase:
        idx2bi[idx] = i + j
        bi2idx[i + j] = idx
        idx += 1


def bi2id(char):
    if char in bi2idx.keys():
        return bi2idx[char]
    else:
        print('Unexpected character: %s' % char)
        return 0


def id2bi(dictid):
    if 0 <= dictid < len(idx2bi):
        return idx2bi[dictid]
    else:
        return '  '


print(bi2id('ad'), bi2id('zf'), bi2id('  '), bi2id('r '), bi2id('ï'))
print(id2bi(31), id2bi(708), id2bi(0), id2bi(486))

batch_size = 64
num_unrollings = 10


class BigramBatchGenerator(object):
    def __init__(self, text, batch_size, num_unrollings):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        segment = self._text_size // batch_size
        # print 'self._text_size, batch_size, segment', self._text_size, batch_size, segment
        self._cursor = [offset * segment for offset in range(batch_size)]
        # print self._cursor
        self._last_batch = self._next_batch()

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
        for b in range(self._batch_size):
            batch[b, bi2id(self._text[self._cursor[b]:self._cursor[b] + 2])] = 1.0
            self._cursor[b] = (self._cursor[b] + 2) % self._text_size
        return batch

    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch]
        for step in range(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches


def characters(probabilities):
    """Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (mostl likely) character representation."""
    return [id2bi(c) for c in np.argmax(probabilities, 1)]


def batches2string(batches):
    """Convert a sequence of batches back into their (most likely) string
    representation."""
    s = [''] * batches[0].shape[0]
    for b in batches:
        s = [''.join(x) for x in zip(s, characters(b))]
    return s


train_batches = BigramBatchGenerator(train_text, batch_size, num_unrollings)
valid_batches = BigramBatchGenerator(valid_text, 1, 1)

print(batches2string(train_batches.next()))
print(batches2string(train_batches.next()))
print(batches2string(valid_batches.next()))
print(batches2string(valid_batches.next()))


def logprob(predictions, labels):
    # prevent negative probability
    """Log-probability of the true labels in a predicted batch."""
    predictions[predictions < 1e-10] = 1e-10
    return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]


def sample_distribution(distribution):
    """Sample one element from a distribution assumed to be an array of normalized
    probabilities.
    """
    # 取一部分数据用于评估，所取数据比例随机
    r = random.uniform(0, 1)
    s = 0
    for i in range(len(distribution)):
        s += distribution[i]
        if s >= r:
            return i
    return len(distribution) - 1


def sample(prediction):
    """Turn a (column) prediction into 1-hot encoded samples."""
    p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
    p[0, sample_distribution(prediction[0])] = 1.0
    return p


def random_distribution():
    """Generate a random column of probabilities."""
    b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
    return b / np.sum(b, 1)[:, None]


num_nodes = 64

graph = tf.Graph()
with graph.as_default():
    # Parameters:
    # Input, Forget, Memory, Output gate: input, previous output, and bias.
    ifcox = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes * 4], -0.1, 0.1))
    ifcom = tf.Variable(tf.truncated_normal([num_nodes, num_nodes * 4], -0.1, 0.1))
    ifcob = tf.Variable(tf.zeros([1, num_nodes * 4]))

    # Variables saving state across unrollings.
    saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    # Classifier weights and biases.
    w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
    b = tf.Variable(tf.zeros([vocabulary_size]))


    def _slice(_x, n, dim):
        return _x[:, n * dim:(n + 1) * dim]


    # Definition of the cell computation.
    def lstm_cell(i, o, state):

        ifco_gates = tf.matmul(i, ifcox) + tf.matmul(o, ifcom) + ifcob

        input_gate = tf.sigmoid(_slice(ifco_gates, 0, num_nodes))
        forget_gate = tf.sigmoid(_slice(ifco_gates, 1, num_nodes))
        update = _slice(ifco_gates, 2, num_nodes)
        state = forget_gate * state + input_gate * tf.tanh(update)
        output_gate = tf.sigmoid(_slice(ifco_gates, 3, num_nodes))
        return output_gate * tf.tanh(state), state


    # Input data.
    train_data = list()
    for _ in range(num_unrollings + 1):
        train_data.append(
            tf.placeholder(tf.float32, shape=[batch_size, vocabulary_size]))

    train_inputs = train_data[:num_unrollings]
    train_labels = train_data[1:]  # labels are inputs shifted by one time step.
    # print('#######', train_inputs)
    # print('#######', train_labels)

    # Unrolled LSTM loop.
    outputs = list()
    output = saved_output
    state = saved_state
    for i in train_inputs:
        output, state = lstm_cell(i, output, state)
        outputs.append(output)

    # State saving across unrollings.
    with tf.control_dependencies([saved_output.assign(output),
                                  saved_state.assign(state)]):
        # Classifier.
        logits = tf.nn.xw_plus_b(tf.concat(outputs, 0), w, b)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=tf.concat(train_labels, 0)))

    # Optimizer.
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        10.0, global_step, 5000, 0.1, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    optimizer = optimizer.apply_gradients(
        zip(gradients, v), global_step=global_step)

    # Predictions.
    train_prediction = tf.nn.softmax(logits)

    # Sampling and validation eval: batch 1, no unrolling.
    sample_input = tf.placeholder(tf.float32, shape=[1, vocabulary_size])
    saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
    saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
    reset_sample_state = tf.group(
        saved_sample_output.assign(tf.zeros([1, num_nodes])),
        saved_sample_state.assign(tf.zeros([1, num_nodes])))
    sample_output, sample_state = lstm_cell(
        sample_input, saved_sample_output, saved_sample_state)
    with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                  saved_sample_state.assign(sample_state)]):
        sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))

num_steps = 7001
summary_frequency = 100

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    mean_loss = 0
    for step in range(num_steps):
        batches = train_batches.next()
        feed_dict = dict()
        for i in range(num_unrollings + 1):
            feed_dict[train_data[i]] = batches[i]
        _, l, predictions, lr = session.run(
            [optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
        mean_loss += l
        if step % summary_frequency == 0:
            if step > 0:
                mean_loss /= summary_frequency
            # The mean loss is an estimate of the loss over the last few batches.
            print(
                'Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
            mean_loss = 0
            labels = np.concatenate(list(batches)[1:])
            print('Minibatch perplexity: %.2f' % float(
                np.exp(logprob(predictions, labels))))
            if step % (summary_frequency * 10) == 0:
                # Generate some samples.
                print('=' * 80)
                for _ in range(5):
                    feed = sample(random_distribution())
                    sentence = characters(feed)[0]
                    reset_sample_state.run()
                    for _ in range(79):
                        prediction = sample_prediction.eval({sample_input: feed})
                        feed = sample(prediction)
                        sentence += characters(feed)[0]
                    print(sentence)
                print('=' * 80)
            # Measure validation set perplexity.
            reset_sample_state.run()
            valid_logprob = 0
            for _ in range(valid_size):
                b = valid_batches.next()
                predictions = sample_prediction.eval({sample_input: b[0]})
                valid_logprob = valid_logprob + logprob(predictions, b[1])
            print('Validation set perplexity: %.2f' % float(np.exp(
                valid_logprob / valid_size)))
