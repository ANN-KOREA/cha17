import numpy as np
import random
import math
from functools import partial

from utils.transformation_3d import affine_transform, apply_affine_transform


DEFAULT_AUGMENTATION_PARAMETERS = {
    "scale": [1, 1, 1],  # factor
    "uniform scale": 1, # factor, equal scale in all directions
    "rotation": [0, 0, 0],  # degrees
    "shear": [0, 0, 0],  # degrees
    "translation": [0, 0, 0],
    "reflection": [0, 0, 0] #Bernoulli p
}


def augment_3d(input_shape, augment_params, output_shape=None, center=None, augm_scale=None, output_scale=None, **kwargs):
    input_shape = np.asarray(input_shape, np.float)

    if output_shape is None: output_shape = input_shape
    else: output_shape = np.asarray(output_shape, np.float)

    if augm_scale is None: augm_scale = np.ones((3,), np.float)
    else: augm_scale = np.asarray(augm_scale, np.float)

    if output_scale is None: output_scale = output_shape/input_shape
    else: output_scale = np.asarray(output_scale, np.float)*output_shape/input_shape

    if center is None: center = input_shape / 2. - 0.5

    shift_center = affine_transform(translation=-center)
    augm_scale = affine_transform(scale=augm_scale)
    augments = affine_transform(**augment_params)
    output_scale = affine_transform(scale=output_scale)
    unshift_center = affine_transform(translation=output_shape / 2. - 0.5)

    tr_matrix = shift_center.dot(augm_scale).dot(augments).dot(output_scale).dot(unshift_center)

    return partial(apply_affine_transform, matrix=tr_matrix, output_shape=output_shape, **kwargs)


def sample_augm_params(augm):
    if augm is None: augm = {}
    new_augm = dict(DEFAULT_AUGMENTATION_PARAMETERS)

    if "scale" in augm:
        new_augm["scale"] = [log_uniform(v) for v in augm["scale"]]
    if "uniform scale" in augm:
        uscale = log_uniform(augm["uniform scale"])
        new_augm["scale"] = [v*uscale for v in new_augm["scale"]]
    if "rotation" in augm:
        new_augm["rotation"] = [uniform(v) for v in augm["rotation"]]
    if "shear" in augm:
        new_augm["shear"] = [uniform(v) for v in augm["shear"]]
    if "translation" in augm:
        new_augm["translation"] = [uniform(v) for v in augm["translation"]]
    if "reflection" in augm:
        new_augm["reflection"] = [bernoulli(v) for v in augm["reflection"]]
    del new_augm["uniform scale"]
    return new_augm


def log_uniform(max_val):
    return math.exp(uniform(math.log(max_val)))


def uniform(max_val):
    return max_val*(random.random()*2-1)


def bernoulli(p): return random.random() < p  #range [0.0, 1.0)