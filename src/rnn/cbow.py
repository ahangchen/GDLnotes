import zipfile
import tensorflow as tf
import numpy as np
import random
import math
import collections

from matplotlib import pylab
from sklearn.manifold import TSNE

from not_mnist.img_pickle import save_obj, load_pickle
from not_mnist.load_data import maybe_download


def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


def build_dataset(words, vocabulary_size):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    context_size = 2 * skip_window
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.float32)
    batchs = np.ndarray(shape=(context_size, batch_size), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    # use data of batch_size to create train_data-label set of batch_size // num_skips * num_skips
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        for j in range(num_skips):
            labels[i * num_skips + j, 0] = buffer[target]
            met_target = False
            for bj in range(context_size):
                if bj == target:
                    met_target = True
                if met_target:
                    batchs[bj, i * num_skips + j] = buffer[bj + 1]
                else:
                    batchs[bj, i * num_skips + j] = buffer[bj]

        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # print('generate batch')
    # print(batchs)
    return batchs, labels

vocabulary_size = 50000
data_set = load_pickle('text8_data.pickle')
if data_set is None:
    # load data
    url = 'http://mattmahoney.net/dc/'
    filename = maybe_download('text8.zip', 31344016, url=url)

    # read data
    words = read_data(filename)
    print('Data size %d' % len(words))
    data, count, dictionary, reverse_dictionary = build_dataset(words, vocabulary_size)
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10])
    del words  # Hint to reduce memory.
    data_set = {
        'data': data, 'count': count, 'dictionary': dictionary, 'reverse_dictionary': reverse_dictionary,
    }
    save_obj('text8_data.pickle', data_set)
else:
    data = data_set['data']
    count = data_set['count']
    dictionary = data_set['dictionary']
    reverse_dictionary = data_set['reverse_dictionary']

# split data
data_index = 0

print('data:', [reverse_dictionary[di] for di in data[:8]])

for num_skips, skip_window in [(2, 1), (4, 2)]:
    test_size = 8
    batch, labels = generate_batch(batch_size=test_size, num_skips=num_skips, skip_window=skip_window)
    print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
    print('    batch:', [reverse_dictionary[bi] for bi in batch.reshape(-1)])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(-1)])

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64  # Number of negative examples to sample.

# tensor: Train a skip-gram model, word2vec
graph = tf.Graph()

with graph.as_default():
    # Input data.
    train_dataset = tf.placeholder(tf.int32, shape=[2 * skip_window, batch_size])
    train_labels = tf.placeholder(tf.float32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, shape=[2 * skip_window, batch_size], dtype=tf.int32)

    # Variables.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    softmax_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Model.
    # Look up embeddings for inputs.
    embed = tf.nn.embedding_lookup(embeddings, train_dataset)
    # sum up vectors on first dimensions, as context vectors
    embed_sum = tf.reduce_sum(embed, 0)

    # Compute the softmax loss, using a sample of the negative labels each time.
    loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases,
                                   train_labels, embed_sum,
                                   num_sampled, vocabulary_size))

    # Optimizer.
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

    # Compute the similarity between minibatch examples and all embeddings.
    # We use the cosine distance:
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    # sum up vectors
    valid_embeddings_sum = tf.reduce_sum(valid_embeddings, 0)
    similarity = tf.matmul(valid_embeddings_sum, tf.transpose(normalized_embeddings))

# flow
num_steps = 100001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    average_loss = 0
    for step in range(num_steps):
        batch_data, batch_labels = generate_batch(
            batch_size, num_skips, skip_window)
        # print(batch_data.shape)
        # print(batch_labels.shape)
        feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += l
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step %d: %f' % (step, average_loss))
            average_loss = 0
        # note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)
    final_embeddings = normalized_embeddings.eval()
    save_obj('text8_embed.pickle', final_embeddings)

num_points = 400

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points + 1, :])


def plot(embeddings, labels):
    assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
    pylab.figure(figsize=(15, 15))  # in inches
    for i, label in enumerate(labels):
        x, y = embeddings[i, :]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                       ha='right', va='bottom')
    pylab.show()


words = [reverse_dictionary[i] for i in range(1, num_points + 1)]
plot(two_d_embeddings, words)
