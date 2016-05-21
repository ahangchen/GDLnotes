import cPickle as pickle
import os
import numpy as np

from not_mnist.img_pickle import load_pickle, save_obj

image_size = 28  # Pixel width and height.


def img_diff(pix_s1, pix_s2):  # by pixels
    dif_cnt = 0
    height = image_size
    width = image_size
    total = width * height
    for x in range(height):
        for y in range(width):
            if pix_s1[x][y] != pix_s2[x][y]:
                dif_cnt += 1
    return float(dif_cnt) / float(total)


def test_img_diff():
    img1 = [[x for x in range(20)] for y in range(28)]
    img2 = [[x for x in range(20)] for y in range(28)]
    print(img_diff(img1, img2))


def img_in(img, imgs):
    for i, img2 in enumerate(imgs):
        if img_diff(img, img2) < 0.1:
            return True
    return False


def BKDRHash(string):
    seed = 131
    hash = 0
    for ch in string:
        hash = hash * seed + ord(ch)
    return hash & 0x7FFFFFFF


def img_hash(pix_s):
    seed = 131
    v_hash = 0
    for row in pix_s:
        for p in row:
            v_hash = v_hash * seed + int(p * 255)
    return v_hash & 0x7FFFFFFF


def imgs_except(left, right):
    return filter(lambda img: not img_in(img, right), left)


def test_imgs_diff():
    img1 = [[x for x in range(20)] for y in range(28)]
    img2 = [[x for x in range(20)] for y in range(28)]
    img3 = [[x for x in range(20)] for y in range(28)]

    print(len(imgs_except([img2, img3], [img1])))


def imgs_idx_except(left, right):
    except_idxs = []
    imgs = []
    for i in range(len(left)):
        print('compare left[%d] to right' % i)
        # about 2-3 seconds for one compare between left[i] and all right
        if img_in(left[i], right):
            except_idxs.append(i)
            imgs.append(left[i])
    return except_idxs, imgs


def imgs_idx_hash_except(left, right):
    except_idxs = []
    right_hashes = [img_hash(img) for img in right]
    print len(right_hashes)
    for i in range(len(left)):
        if img_hash(left[i]) in right_hashes:
            print('compare left[%d] to right found the same' % i)
            except_idxs.append(i)
    res = np.delete(left, except_idxs, axis=0)
    return except_idxs, res


def list_except(objs, idxs):
    new_objs = []
    for i in range(len(objs)):
        if i not in idxs:
            new_objs.append(objs[i])
    return new_objs


def clean():
    datasets = load_pickle('notMNIST.pickle')
    test_dataset = datasets['test_dataset']
    test_labels = datasets['test_labels']
    print('test_dataset:%d' % len(test_dataset))
    print('test_labels:%d' % len(test_labels))

    except_valid_idx, valid_dataset = imgs_idx_hash_except(datasets['valid_dataset'], test_dataset)
    valid_labels = np.delete(datasets['valid_labels'], except_valid_idx)
    print('valid_dataset:%d' % len(valid_dataset))
    print('valid_labels:%d' % len(valid_labels))

    # except with valid_dataset
    except_train_idx, train_dataset = imgs_idx_hash_except(datasets['train_dataset'], valid_dataset)
    train_labels = np.delete(datasets['train_labels'], except_train_idx)
    # except with test_dataset
    except_train_idx, train_dataset = imgs_idx_hash_except(train_dataset, test_dataset)
    train_labels = np.delete(train_labels, except_train_idx)

    print('train_dataset:%d' % len(train_dataset))
    print('train_labels:%d' % len(train_labels))
    print('valid_dataset:%d' % len(valid_dataset))
    print('valid_labels:%d' % len(valid_labels))
    print('test_dataset:%d' % len(test_dataset))
    print('test_labels:%d' % len(test_labels))

    pickle_file = 'notMNIST_clean.pickle'
    save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
    }
    save_obj(pickle_file, save)


if __name__ == '__main__':
    clean()
