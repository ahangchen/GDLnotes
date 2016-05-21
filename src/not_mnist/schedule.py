from not_mnist.clean_overlap import clean
from not_mnist.extract import maybe_extract
from not_mnist.img_pickle import maybe_pickle, save_obj
from not_mnist.load_data import maybe_download
from not_mnist.logistic_train import load_train
from not_mnist.merge_prune import merge_datasets, randomize, merge_prune

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)

train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)

train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
    train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

merge_prune(train_folders, test_folders)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)
clean()
load_train()
