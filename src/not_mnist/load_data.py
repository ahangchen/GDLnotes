from __future__ import print_function

import os
import sys

# from six.moves.urllib.request import urlretrieve
# from six.moves import cPickle as pickle
from urllib import urlretrieve

# %matplotlib inline

# url = 'http://commondatastorage.googleapis.com/books1000/'
# if the url above can't work, use this:

last_percent_reported = None

# First, we'll download the dataset to our local machine.
# The data consists of characters rendered in a variety of fonts on a 28x28 image.
# The labels are limited to 'A' through 'J' (10 classes).
# The training set has about 500k and the testset 19000 labelled examples.
# Given these sizes, it should be possible to train models quickly on any machine.


def download_progress_hook(count, blockSize, totalSize):
    """A hook to report the progress of a download. This is mostly intended for users with
    slow internet connections. Reports every 1% change in download progress.
    """
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        last_percent_reported = percent


def maybe_download(filename, expected_bytes, url='https://commondatastorage.googleapis.com/books1000/', force=False, ):
    """Download a file if not present, and make sure it's the right size."""
    if force or not os.path.exists(filename):
        print('Attempting to download:', filename)
        filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
        print('\nDownload Complete!')
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


if __name__ == '__main__':
    train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
    test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

