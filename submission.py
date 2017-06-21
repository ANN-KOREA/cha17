import numpy as np
import cPickle
import argparse
from glob import glob

from utils import TRAIN, VALID, secs_to_str
import paths
import string
import zipfile
import scipy.signal
import scipy.stats

def main():

    args = parse_args()
    set_ = args.set
    meta = cPickle.load(open(args.meta, "rb"))
    cfg_name = meta["configuration"]
    expid = meta["experiment_id"]+"--"+set_

    pred_dir = "./predictions/"
    s_preds = cPickle.load(open(pred_dir+expid+".pkl", "rb"))

    vid_paths = glob(paths.CON_PREP2+"*/*/*")
    vid_paths.sort()
    set_vid_idxs = [i for i, p in enumerate(vid_paths) if set_ in p]
    print len(set_vid_idxs), set_, "videos"

    for i, s_pred in enumerate(s_preds):
        # print s_pred
        # print scipy.signal.medfilt(s_pred, [19])
        # print
        # s_preds[i] = scipy.signal.medfilt(s_pred, [19]).astype("int32")
        s_preds[i] = modefilter(s_pred, 35).astype("int32")
        # print s_pred
        # print s_preds[i]
        # print

    if set_ == VALID:
        framewise_class = {}
        for i, vid_idx in enumerate(set_vid_idxs):
            framewise_class[vid_idx] =  np.zeros((len(s_preds[i]), 249), "float32")

        _vid_paths = {}
        for i, p in enumerate(vid_paths):
            if set_ in p:
                _vid_paths[string.join(p.split("/")[-2:], "/")] = (i, p)


        with open(paths.CON_LBL_VALID, "r") as f: lines = f.readlines()

        for line in lines:
            line = line.split(" ")
            vid_id, vid_path = _vid_paths[line[0]]
            assert line[1].startswith("1")
            for smpl in line[1:]:
                begin_end, target = smpl.split(":")
                begin, end = begin_end.split(",")
                begin, end, target = int(begin) - 1, int(end) - 1, int(target) - 1
                framewise_class[vid_id][begin:end, target] = 1

        ji = []
        for i, s_pred in enumerate(s_preds):
            ji.append(calc_ji(s_pred, framewise_class[set_vid_idxs[i]]))

        print "JI", np.mean(ji)


    subm_file = open(pred_dir + expid + ".txt", "w")

    for i, s_pred in enumerate(s_preds):
        line = ""
        path = vid_paths[set_vid_idxs[i]]
        line += string.join(path.split("/")[-2:], "/")
        begin = 1
        prev = None
        for f, p in enumerate(s_pred):
            if prev is None: prev = p
            elif prev != p:
                line += " %i,%i:%i"%(begin, f, prev+1)
                begin = f+1
                prev = p
            elif f == len(s_pred)-1:
                line +=  " %i,%i:%i" % (begin, f+1, prev + 1)

        if len(s_pred) == 0:
            line += " 1,1:1"

        # print line
        # print s_pred
        line += "\n"
        subm_file.write(line)

    subm_file.close()
    zf = zipfile.ZipFile(pred_dir + expid + ".zip", "w", zipfile.ZIP_DEFLATED)
    zf.write(pred_dir + expid + ".txt", arcname="%s_prediction.txt"%set_)
    zf.close()
    # shutil.make_archive(pred_dir + expid + ".txt.zip", 'zip', pred_dir + expid + ".txt")

    # if set_ == VALID
    # with open(gt_path, "r") as f: lines = f.readlines()

def calc_ji(preds, targets):
    p = preds.flatten().astype("int32")
    p = np.eye(249)[p]
    intersection = np.sum(targets * p, axis=0)
    tarsum = np.sum(targets, axis=0)
    if np.count_nonzero(tarsum) == 0: return 0
    union = tarsum + np.sum(p, axis=0) - intersection
    ji = intersection / (union + 1e-9)
    return ji.sum() / np.count_nonzero(tarsum)


def modefilter(array, kernel):
    n_array = np.empty(array.shape, array.dtype)
    for i in range(len(array)):
        begin = max(0, i - kernel // 2)
        end = min(begin + kernel, len(array))
        begin = max(0, end-kernel)
        n_array[i] = scipy.stats.mode(array[begin:end])[0]
    return n_array

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=str, default="metadata/default.pkl")
    parser.add_argument("--set", type=str, default=VALID)

    args, unknown = parser.parse_known_args()
    print args
    return args


if __name__ == '__main__':
    main()