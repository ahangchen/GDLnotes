from tensorflow.examples.tutorials.mnist import input_data

from not_mnist.pick import save_obj
from not_mnist.pick import load_pickle
import numpy as np

mnist_path = 'mnist'


def img_reshape(data, length):
    img_size = 28
    depth = 1
    # print(len(data))
    # print(length)
    return np.array(data).reshape(length, img_size, img_size, depth)


def label_reshape(data, length):
    label_size = 10
    # print(len(data))
    # print(length)
    return np.array(data).reshape(length, label_size)


def format_mnist():
    mnist = load_pickle(mnist_path)
    if mnist is None:
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        save_obj(mnist_path, mnist)
    train_length = len(mnist.train.labels)
    valid_length = len(mnist.validation.labels)
    test_length = len(mnist.test.labels)
    return img_reshape(mnist.train.images, train_length), label_reshape(mnist.train.labels, train_length), \
           img_reshape(mnist.validation.images, valid_length), label_reshape(mnist.validation.labels, valid_length), \
           img_reshape(mnist.test.images, test_length), label_reshape(mnist.test.labels, test_length)


if __name__ == '__main__':
    format_mnist()
