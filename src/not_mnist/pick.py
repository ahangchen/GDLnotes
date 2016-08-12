import os

import cPickle as pickle


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


def load_pickle(pickle_name):
    # load a pickle file to memory
    if os.path.exists(pickle_name):
        return pickle.load(open(pickle_name, "r"))
    return None
