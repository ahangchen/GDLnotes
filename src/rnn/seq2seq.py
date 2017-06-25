# coding=utf-8
import math
import string
import zipfile

import numpy as np
import tensorflow as tf
import seq2seq_model

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
valid_size = 100
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
print(train_size, train_text[:64])
print(valid_size, valid_text[:64])

vocabulary_size = 35  # len(string.ascii_lowercase) + 2 # [a-z] + ' '
first_letter = ord(string.ascii_lowercase[0])


def char2id(char):
    if char in string.ascii_lowercase:
        return ord(char) - first_letter + 5
    elif char == ' ':
        return 4
    elif char == '!':
        return 31
    else:
        print('Unexpected character: %s' % char)
        return 0


def id2char(dictid):
    if dictid == 31:
        return '!'
    elif dictid > 4:
        return chr(dictid + first_letter - 5)
    elif dictid == 4:
        return ' '
    else:
        return '@'


print(char2id('a'), char2id('z'), char2id(' '), char2id('!'))
print(id2char(5), id2char(30), id2char(4), id2char(31))
batch_size = 64
num_unrollings = 19


class BatchGenerator(object):
    def __init__(self, text, batch_size, num_unrollings):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        segment = self._text_size // num_unrollings
        self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._next_batch(0)

    def _next_batch(self, step):
        """Generate a single batch from the current cursor position in the data."""
        batch = ''
        # print('text size', self._text_size)
        for b in range(self._num_unrollings):
            # print(self._cursor[step])
            self._cursor[step] %= self._text_size
            batch += self._text[self._cursor[step]]
            self._cursor[step] += 1
        return batch

    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch]
        for step in range(self._batch_size):
            batches.append(self._next_batch(step))
        self._last_batch = batches[-1]
        return batches


def characters(probabilities):
    """Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (most likely) character representation."""
    return [id2char(c) for c in np.argmax(probabilities, 1)]


def ids(probabilities):
    """Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (most likely) character representation."""
    return [str(c) for c in np.argmax(probabilities, 1)]


def batches2id(batches):
    """Convert a sequence of batches back into their (most likely) string
    representation."""
    s = [''] * batches[0].shape[0]
    for b in batches:
        s = [''.join(x) for x in zip(s, ids(b))]
    return s

train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
valid_batches = BatchGenerator(valid_text, 1, num_unrollings)


def rev_id(forward):
    temp = forward.split(' ')
    backward = []
    for i in range(len(temp)):
        backward += temp[i][::-1] + ' '
    return map(lambda x: char2id(x), backward[:-1] + ['!'])


def create_model(sess, forward_only):
    model = seq2seq_model.Seq2SeqModel(source_vocab_size=vocabulary_size,
                                       target_vocab_size=vocabulary_size,
                                       buckets=[(20, 21)],
                                       size=256,
                                       num_layers=4,
                                       max_gradient_norm=5.0,
                                       batch_size=batch_size,
                                       learning_rate=1.0,
                                       learning_rate_decay_factor=0.9,
                                       use_lstm=True,
                                       forward_only=forward_only)
    return model


with tf.Session() as sess:
    model = create_model(sess, False)
    sess.run(tf.global_variables_initializer())
    num_steps = 30001

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    step_ckpt = 100
    valid_ckpt = 500

    for step in range(1, num_steps):
        model.batch_size = batch_size
        train_batches_next = train_batches.next()
        batches = train_batches_next
        train_sets = []
        batch_encs = map(lambda x: map(lambda y: char2id(y), list(x)), batches)
        batch_decs = map(lambda x: rev_id(x), batches)
        for i in range(len(batch_encs)):
            train_sets.append((batch_encs[i], batch_decs[i]))

        # Get a batch and make a step.
        encoder_inputs, decoder_inputs, target_weights = model.get_batch([train_sets], 0)
        _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, 0, False)

        loss += step_loss / step_ckpt

        # Once in a while, we save checkpoint, print statistics, and run evals.
        if step % step_ckpt == 0:
            # Print statistics for the previous epoch.
            perplexity = math.exp(loss) if loss < 300 else float('inf')
            print ("global step %d learning rate %.4f perplexity "
                   "%.2f" % (model.global_step.eval(), model.learning_rate.eval(), perplexity))
            # Decrease learning rate if no improvement was seen over last 3 times.
            if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                sess.run(model.learning_rate_decay_op)
            previous_losses.append(loss)

            loss = 0.0

            if step % valid_ckpt == 0:
                v_loss = 0.0

                model.batch_size = 1
                batches = ['the quick brown fox']
                test_sets = []
                batch_encs = map(lambda x: map(lambda y: char2id(y), list(x)), batches)
                # batch_decs = map(lambda x: rev_id(x), batches)
                test_sets.append((batch_encs[0], []))
                # Get a 1-element batch to feed the sentence to the model.
                encoder_inputs, decoder_inputs, target_weights = model.get_batch([test_sets], 0)
                # Get output logits for the sentence.
                _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, 0, True)

                # This is a greedy decoder - outputs are just argmaxes of output_logits.
                outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
                # If there is an EOS symbol in outputs, cut them at that point.

                if char2id('!') in outputs:
                    outputs = outputs[:outputs.index(char2id('!'))]

                print('>>>>>>>>> ', batches[0], ' -> ', ''.join(map(lambda x: id2char(x), outputs)))

                for _ in range(valid_size):
                    model.batch_size = 1
                    v_batches = valid_batches.next()
                    valid_sets = []
                    v_batch_encs = map(lambda x: map(lambda y: char2id(y), list(x)), v_batches)
                    v_batch_decs = map(lambda x: rev_id(x), v_batches)
                    for i in range(len(v_batch_encs)):
                        valid_sets.append((v_batch_encs[i], v_batch_decs[i]))
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch([valid_sets], 0)
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, 0, True)
                    v_loss += eval_loss / valid_size

                eval_ppx = math.exp(v_loss) if v_loss < 300 else float('inf')
                print("  valid eval:  perplexity %.2f" % (eval_ppx))

    # reuse variable -> subdivide into two boxes
    model.batch_size = 1  # We decode one sentence at a time.
    batches = ['the quick brown fox']
    test_sets = []
    batch_encs = map(lambda x: map(lambda y: char2id(y), list(x)), batches)
    # batch_decs = map(lambda x: rev_id(x), batches)
    test_sets.append((batch_encs[0], []))
    # Get a 1-element batch to feed the sentence to the model.
    encoder_inputs, decoder_inputs, target_weights = model.get_batch([test_sets], 0)
    # Get output logits for the sentence.
    _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, 0, True)
    # This is a greedy decoder - outputs are just argmaxes of output_logits.
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    print ('## : ', outputs)
    # If there is an EOS symbol in outputs, cut them at that point.
    if char2id('!') in outputs:
        outputs = outputs[:outputs.index(char2id('!'))]

    print(batches[0], ' -> ', ''.join(map(lambda x: id2char(x), outputs)))
