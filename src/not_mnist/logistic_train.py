from __future__ import print_function

import os

from sklearn.linear_model import LogisticRegression

from not_mnist.img_pickle import load_pickle, save_obj


def load_train():
    datasets = load_pickle('notMNIST_clean.pickle')
    train_dataset = datasets['train_dataset']
    train_labels = datasets['train_labels']
    valid_dataset = datasets['valid_dataset']
    valid_labels = datasets['valid_labels']

    classifier_name = 'classifier.pickle'

    if os.path.exists(classifier_name):
        classifier = load_pickle(classifier_name)
    else:
        classifier = LogisticRegression()
        classifier.fit(train_dataset.reshape(train_dataset.shape[0], -1), train_labels)
        save_obj(classifier_name, classifier)

    # simple valid
    valid_idx_s = 3000
    valid_idx_e = 3014
    x = classifier.predict(valid_dataset.reshape(valid_dataset.shape[0], -1)[valid_idx_s: valid_idx_e])
    print(x)
    print(valid_labels[valid_idx_s:valid_idx_e])

    # whole valid
    x = classifier.predict(valid_dataset.reshape(valid_dataset.shape[0], -1))
    fail_cnt = 0
    for i, pred in enumerate(x):
        if pred != valid_labels[i]:
            fail_cnt += 1
    print("success rate:" + str((1 - float(fail_cnt) / len(x)) * 100) + "%")

if __name__ == '__main__':
    load_train()
