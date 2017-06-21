import numpy as np
from functools import partial

from utils.augmentation_3d import augment_3d, sample_augm_params,DEFAULT_AUGMENTATION_PARAMETERS
from utils import VALID, TEST


class Augment3D(object):
    def __init__(self, output_shape, augm_params=None,
                 augm_scale=None, output_scale=None, interp_order=1, mode="constant", cval=0, dbias=0):
        """mode: 'constant', 'nearest', 'reflect' or 'wrap'"""
        self.augm_params = augm_params
        self.output_shape = output_shape
        self.dbias = dbias
        self.augment = partial(augment_3d, output_shape=output_shape,augm_scale=augm_scale, output_scale=output_scale,
            order=interp_order, mode=mode, cval=cval)

    def process(self, sample):
        set_ = sample["set"]
        if set_ == VALID or set_ == TEST:
            augm_params = dict(DEFAULT_AUGMENTATION_PARAMETERS)
            del augm_params["uniform scale"]
        else:
            augm_params = sample_augm_params(self.augm_params)

        vid = sample["video"]

        if vid.ndim == 4:
            apply_transform = self.augment(vid[...,0].shape, augm_params)
            new_vid = np.empty(self.output_shape+(vid.shape[-1],), vid.dtype)

            for channel in range(vid.shape[-1]):
                new_vid[...,channel] = apply_transform(vid[...,channel])

            if self.dbias != 0:
                new_vid[..., -1] += self.dbias
        else:
            apply_transform = self.augment(vid.shape, augm_params)
            new_vid = apply_transform(vid)

            if self.dbias != 0:
                new_vid += self.dbias


        sample["video"] = new_vid