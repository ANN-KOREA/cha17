import numpy as np
import random
from collections import defaultdict

from utils import resample_order0

class ClassPerFrame(object):

    def __init__(self, n_frames):
        self.n_frames = n_frames

    def prepare(self, loader):
        lbls = loader.labels
        framewise_lbls = defaultdict(list)

        for lbl in lbls:
            vid_id = lbl["vid_id"]
            end = lbl["end"]
            framewise_lbls[vid_id].extend([lbl["target"]]*(end-lbl["begin"]+1))
            framewise_lbls[vid_id] = framewise_lbls[vid_id][:end + 1]
            if end != len(framewise_lbls[vid_id]) - 1:
                print len(framewise_lbls[vid_id]), end, lbl["path"]

        self.framewise_lbls = framewise_lbls

        # DEBUG!
        # random.shuffle(lbls)

        loader.labels = lbls


    def process(self, sample):
        begin = sample["begin"]
        end = sample["end"]
        vid_id = sample["vid_id"]

        targets = self.framewise_lbls[vid_id][begin:end]

        if len(targets) != self.n_frames:
            targets = resample_order0(targets, self.n_frames)

        sample["target"] = np.array(targets, "int32")

