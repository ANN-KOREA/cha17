import os
import errno
import cPickle
import numpy as np

TRAIN = "train"
VALID = "valid"
TEST = "test"
INPUT = "input"
OUTPUT = "output"


def is_sorted(x, key = lambda x: x): return all([key(x[i]) <= key(x[i + 1]) for i in xrange(len(x) - 1)])

class ArrayDummy(object):
    shape = ()
    dtype = "float32"


def arr(shape=(), dtype="float32"):
    array = ArrayDummy()
    array.shape = shape
    array.dtype = dtype
    return array


def indexing(list_, indices):
    return [list_[i] for i in indices]


def argsort(list_):
    return [i[0] for i in sorted(enumerate(list_), key=lambda x: x[1])]


def remove_indices(list_, indices):
    return [el for i, el in enumerate(list_) if i not in indices]


intr = lambda x: int(round(x))


def secs_to_str(t): return "%dh%02dm" % reduce(lambda ll,b : divmod(ll[0],b) + ll[1:],[(t*1000,),1000,60,60])[:2]


def normalize_minmax(x, new_min=0., new_max=1.):
    old_min = x.min()
    return (x - old_min) * (new_max - new_min) / float((x.max() - old_min) + new_min)


def normalize(x, old_min, old_max, new_min=0., new_max=1.):
    return (x - old_min) * (new_max - new_min) / float((old_max - old_min) + new_min)


def make_sure_path_exists(path):
    """Try to create the directory, but if it already exist we ignore the error"""
    try: os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST: raise


def remove_file_if_exists(path):
    if os.path.exists(path): os.remove(path)


def save_pkl(obj, path, protocol=cPickle.HIGHEST_PROTOCOL):
    with file(path, 'wb') as f:
        cPickle.dump(obj, f, protocol=protocol)


def load_pkl(path):
    with file(path, 'rb') as f:
        obj = cPickle.load(f)
    return obj


def invert_dict(d):
    return {v: k for k, v in d.iteritems()}


def onehot(x, n_values): return np.eye(n_values)[x]

RGB2GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114], np.float32)
def rgb2gray(rgb):
    return np.dot(rgb, RGB2GRAY_WEIGHTS)


def resample_order0(a, ln):
    l = len(a)
    ln1 = float(ln-1)
    l1 = float(l-1)
    return [ a[int(round(i/ln1*l1))] for i in range(ln) ]




class struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
    def __repr__(self):
        return '{%s}' % str(', '.join('%s : %s' % (k, repr(v)) for
      (k, v) in self.__dict__.iteritems()))
    def keys(self):
        return self.__dict__.keys()