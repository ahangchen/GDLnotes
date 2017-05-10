# coding=utf-8
import os
import random

from util.file_helper import read_lines

caltch_path = '/Users/cwh/Mission/scut/lesson/PR/project/PR_dataset/'
caltech_train_path = caltch_path + 'train/'
caltech_test_path = caltch_path + 'test/'
caltech_train_pos_path = caltech_train_path + 'pos/'
caltech_train_neg_path = caltech_train_path + 'neg/'
caltech_test_pos_path = caltech_test_path + 'pos/'
caltech_test_neg_path = caltech_test_path + 'neg/'


def read_caltech():
    train_features = list()
    train_labels = list()
    test_features = list()
    test_labels = list()
    i = 0
    max_read = 15000
    for files in os.listdir(caltech_train_pos_path):
        if i > max_read:
            break
        i += 1
        path = os.path.join(caltech_train_pos_path, files)
        features = read_lines(path)
        # print(path)
        train_features.append(features)
        train_features.append(features)
        train_features.append(features)
        train_labels.append([1, 0])
        train_labels.append([1, 0])
        train_labels.append([1, 0])
    pos_train_cnt = len(train_labels) / 3
    print(pos_train_cnt)
    i = 0
    for files in os.listdir(caltech_train_neg_path):
        if i > max_read:
            break
        i += 1
        path = os.path.join(caltech_train_neg_path, files)
        # print(path)
        features = read_lines(path)
        train_features.append(features)
        train_labels.append([0, 1])
    print(len(train_labels) - pos_train_cnt * 3)
    i = 0
    for files in os.listdir(caltech_test_pos_path):
        if i > max_read:
            break
        i += 1
        path = os.path.join(caltech_test_pos_path, files)
        # print(path)
        features = read_lines(path)
        test_features.append(features)
        test_labels.append([1, 0])
    pos_test_cnt = len(test_labels) / 3
    print(pos_test_cnt)
    i = 0
    for files in os.listdir(caltech_test_neg_path):
        if i > max_read:
            break
        i += 1
        path = os.path.join(caltech_test_neg_path, files)
        # print(path)
        features = read_lines(path)
        test_features.append(features)
        test_labels.append([0, 1])
    print(len(test_labels) - pos_test_cnt)
    train_tuples = zip(train_features, train_labels)
    random.shuffle(train_tuples)
    train_features = [train_tuple[0] for train_tuple in train_tuples]
    train_labels = [train_tuple[1] for train_tuple in train_tuples]
    test_tuples = zip(test_features, test_labels)
    random.shuffle(test_tuples)
    test_features = [test_tuple[0] for test_tuple in test_tuples]
    test_labels = [test_tuple[1] for test_tuple in test_tuples]

    return train_features, train_labels, test_features, test_labels


if __name__ == '__main__':
    read_caltech()
