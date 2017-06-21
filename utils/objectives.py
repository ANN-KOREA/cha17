import theano.tensor as T
import numpy as np


def soft_margin_loss(positive, negative):
    return T.nnet.softplus(-positive) + T.nnet.softplus(negative)


# https://groups.google.com/forum/#!topic/theano-users/4KulqXRy7rs
def max_margin_loss(positive, negative, margin=1.):
    return T.nnet.relu(negative - positive + margin)


def power_mean(x, p, axis=-1):
    return T.pow(T.pow(x, np.float32(p)).mean(axis=axis), np.float32(1./p))


def log_mean_exp(x, r=10., axis=-1, eps=1e-7):
    r = np.float32(r)
    return T.log(T.mean(T.exp(r * x), axis=axis)+eps) / r


def cosine(x, y):
    dot = (x * y).sum(axis=-1)
    norm = T.sqrt(squared_magnitude(x) * squared_magnitude(y))
    cos_sim = dot / (norm+1e-7) # [-1, 1]
    # return (cos_sim + 1.) / 2.
    # return T.clip( (cos_sim + 1.) / 2., 0, 1) # [0,1]
    return cos_sim  # [-1, 1]


def dot_product(x, y):
    return (x * y).sum(axis=-1)


def euclidean(x, y):
    return magnitude(x - y)



def squared_euclidean(x, y):
    return squared_magnitude(x - y)


def squared_magnitude(x):
    return T.sqr(x).sum(axis=-1)


def magnitude(x):
    return T.sqrt(squared_magnitude(x)+1e-7)



