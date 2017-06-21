import string
import random
from collections import defaultdict

import paths
from utils import TRAIN, VALID, TEST


class LabelsCon(object):
    def __init__(self): pass

    def prepare(self, loader):
        lbls = []
        # class_bal = {TRAIN:defaultdict(int), VALID:defaultdict(int)}
        idsperclass = {TRAIN:defaultdict(list), VALID:defaultdict(list)}

        id_ = 0

        for set_, path in [(TRAIN, paths.CON_LBL_TRAIN), (VALID, paths.CON_LBL_VALID)]:
            print set_, path

            vid_paths = {}
            for i, p in enumerate(loader.vid_paths):

                if set_ in p:

                # DEBUG !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # if TRAIN in p:

                    vid_paths[string.join(p.split("/")[-2:], "/")] = (i, p)


            with open(path, "r") as f: lines = f.readlines()

            # DEBUG !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # with open(paths.CON_LBL_TRAIN, "r") as f: lines = f.readlines()
            # print len(lines)
            # if set_ == VALID: lines = lines[11000:]
            # else: lines = lines[:11000]

            # o_len_lbls = len(lbls)

            for line in lines:
                line = line.split(" ")
                vid_id, vid_path = vid_paths[line[0]]
                assert line[1].startswith("1")
                for smpl in line[1:]:
                    begin_end, target = smpl.split(":")
                    begin, end = begin_end.split(",")
                    begin, end, target = int(begin)-1, int(end)-1, int(target)-1
                    lbls.append({
                        "path": vid_path,
                        "vid_id": vid_id,
                        "begin": begin,
                        "end": end,
                        "target": target
                    })
                    # class_bal[set_][target] += 1
                    # id_ = len(loader.ids[set_])
                    idsperclass[set_][target].append(id_)
                    loader.ids[set_].append(id_)
                    id_ += 1
            # loader.ids[set_] = range(o_len_lbls, len(lbls))
            print " ", len(loader.ids[set_]), set_, "samples"


        # for i in range(249):
        #     print class_bal[TRAIN][i]/float(len(loader.ids[TRAIN])), class_bal[VALID][i]/float(len(loader.ids[VALID]))

        for id_ in loader.ids[VALID]:
            assert VALID in lbls[id_]["path"]
            assert id_ not in loader.ids[TRAIN]

        for id_ in loader.ids[TRAIN]:
            assert TRAIN in lbls[id_]["path"]
            assert id_ not in loader.ids[VALID]

        loader.labels = lbls
        loader.idsperclass = idsperclass


