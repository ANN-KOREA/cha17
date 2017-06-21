import random
import numpy.random as npr
from glob import glob
from moviepy.editor import VideoFileClip

from utils import TRAIN, VALID, TEST
from utils.baseloader import BaseLoader
import paths

TEST_NUM = 100
VALID_NUM = 100


class LoaderCon(BaseLoader):
    def __init__(self, data_path=paths.CON_PREP, balance=False, *args, **kwargs):
        super(LoaderCon, self).__init__(*args, **kwargs)
        self.data_path = data_path
        self.balance = balance

    def prepare(self):
        # random.seed("seed")
        # self.rng = npr.RandomState(1337)
        self.voc = {}
        self.ids = {TRAIN:[], VALID:[], TEST:[]}
        self.labels = []

        self.vid_paths = glob(self.data_path+"*/*/*")
        self.vid_paths.sort()

    def load_sample(self, id_):
        for set_ in (TEST, VALID, TRAIN):
            if id_ in self.ids[set_]: break

        sample = {"id": id_, "set": set_}
        sample.update(self.labels[id_]) # (path, begin, end, target)
        return sample

    def sample_ids(self, set_, size, offset):
        if set_ == TRAIN:
            # DEBUG
            # return [self.ids[set][49], self.ids[set][24]]* (size/2)
            # return [self.ids[set][49]]*size

            # return self.rng.choice(self.ids[set_], size=size, replace=False)

            # print ids

            if self.balance:
                classes = npr.choice(range(249), size=size, replace=False)
                ids = [npr.choice(self.idsperclass[set_][c]) for c in classes]
            else:
                offset = offset % len(self.ids[set_])
                assert offset + size <= len(self.ids[set_])
                ids = self.ids[set_][offset:offset + size]

            return ids
        else:

            state = npr.get_state()
            npr.seed(168465+offset)

            ids = npr.choice(self.ids[set_], size=size, replace=False)
            # classes = npr.choice(range(249), size=size, replace=False)
            # ids = [npr.choice(self.idsperclass[set_][c]) for c in classes]
            # print ids

            npr.set_state(state)
            return ids

            # return self.ids[set_][offset:offset + size]

