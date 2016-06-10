from __future__ import print_function

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
# from six.moves.urllib.request import urlretrieve
# from six.moves import cPickle as pickle
import cPickle as pickle

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.


def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
    print(folder)
    for image_index, image in enumerate(image_files):
        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) -
                          pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[image_index, :, :] = image_data
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    num_images = image_index + 1
    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names


def show_imgs(imgs, show_max=-1):
    show_cnt = show_max
    if show_max == -1:
        show_cnt = len(imgs)

    for image_index in xrange(show_cnt):
        # they are binary images, if RGBs, don't add cmap="Graeys"
        plt.imshow(imgs[image_index], cmap="Greys")
        plt.show()


def load_pickle(pickle_name):
    # load a pickle file to memory
    if os.path.exists(pickle_name):
        return pickle.load(open(pickle_name, "r"))
    return None


def save_obj(pickle_file, obj):
    try:
        f = open(pickle_file, 'wb')
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
    statinfo = os.stat(pickle_file)
    print('Compressed pickle size:', statinfo.st_size)

if __name__ == '__main__':
    train_folders = ['notMNIST_large/A', 'notMNIST_large/B', 'notMNIST_large/C', 'notMNIST_large/D', 'notMNIST_large/E',
                     'notMNIST_large/F', 'notMNIST_large/G', 'notMNIST_large/H', 'notMNIST_large/I', 'notMNIST_large/J']
    test_folders = ['notMNIST_small/A', 'notMNIST_small/B', 'notMNIST_small/C', 'notMNIST_small/D', 'notMNIST_small/E',
                    'notMNIST_small/F', 'notMNIST_small/G', 'notMNIST_small/H', 'notMNIST_small/I', 'notMNIST_small/J']
    train_datasets = maybe_pickle(train_folders, 45000)
    test_datasets = maybe_pickle(test_folders, 1800)

    for i in range(1): # only load a.pickle
        imgs = load_pickle(train_datasets[i])
        show_imgs(imgs, 3)

