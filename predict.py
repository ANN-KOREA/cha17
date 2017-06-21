import importlib
import sys
import os
import platform
import lasagne as nn
import numpy as np
import theano
import theano.tensor as T
import cPickle
import shutil
import argparse
from glob import glob
from collections import OrderedDict
from time import time, strftime, localtime
# from sklearn.metrics import jaccard_similarity_score

from utils.traintools import detect_nans, TrainLogger, plot_progress, print_model, stopwatch
from utils import TRAIN, VALID, secs_to_str


def main():
###############################################################################################################
# INITIALIZE
###############################################################################################################

    args = parse_args()
    set_ = args.set
    meta = cPickle.load(open(args.meta, "rb"))
    cfg_name = meta["configuration"]
    expid = meta["experiment_id"]+"--"+set_
    params = meta["parameters"]

    cfg = importlib.import_module("models.%s" % cfg_name)

    log = TrainLogger("metadata/%s.log"%expid)
    log.print_to_log()

    print "\n"
    print "EVALUATING", set_, "SET!"
    print "\n"
    print "Model:", cfg_name,  "expid:", expid
    print "batch_size", cfg.batch_size, "batches_per_chunk", cfg.batches_per_chunk, \
        "learning_rate", cfg.learning_rate, "validate_every_n_chunks", cfg.validate_every_n_chunks, \
        "n_valid_chunks", cfg.n_valid_chunks


    data_path = cfg.data_loader.data_path
    print data_path
    vid_meta = cPickle.load(open("./data/vidmeta.pkl", "rb"))
    # print vid_meta

    vid_paths = glob(data_path+"*/*/*")
    vid_paths.sort()
    print len(vid_paths), len(vid_meta)
    set_vid_idxs = [i for i, p in enumerate(vid_paths) if set_ in p]
    print len(set_vid_idxs), set_, "videos"

    # sys.exit()

    cfg.data_loader.predict=True
    cfg.data_loader.start(n_jobs=args.jobs)

###############################################################################################################
# BUILD
###############################################################################################################

    print "Building model ..."
    model = cfg.build_model()
    nn.layers.helper.set_all_param_values(model["output"], params)
    if hasattr(cfg,"is_pretrained") and cfg.is_pretrained: cfg.set_pretrained_params(model)
    all_layers = nn.layers.get_all_layers(model["output"])
    all_params = nn.layers.get_all_params(model["output"], trainable=True)
    print_model(all_layers, all_params)#, {"WeightNormLayer", "LayerNormLayer", "BatchNormLayer", "NonlinearityLayer"})
    model_out = nn.layers.get_output(model["output"], deterministic=True)

    # batch_idx = T.iscalar('idx')
    # data_shared = OrderedDict({tag: nn.utils.shared_empty(len(arr.shape)+1, arr.dtype)
    #                            for (tag, arr) in cfg.data_tags.items()})
    # givens = OrderedDict()
    # for (tag, l) in model["input"].items():
    inp = model["input"]["video"]
    # givens[inp.input_var] = data_shared["video"][batch_idx*cfg.batch_size : (batch_idx+1)*cfg.batch_size]

    # print model["input"].keys()
    print "Compiling evaluation function"
    iter_eval = theano.function([inp.input_var], [model_out])

###############################################################################################################
# PREDICT
###############################################################################################################

    print "Predicting ..."


    import bcolz
    if hasattr(args, "bcolz"):
        bcolz.set_nthreads(args.bcolz)

    chunk_size = 1 #cfg.batch_size * cfg.batches_per_chunk
    cut_off = 8

    preprocessors = cfg.data_loader.preprocessors
    vidprep = None
    classperframe = None
    for p in preprocessors:
        if "VideoLoadPrep" == p.__class__.__name__:
            vidprep = p
        elif "ClassPerFrame" == p.__class__.__name__:
            classperframe = p

    print "Evaluating %s set..." % set_

    stride = cfg.n_frames-cut_off*2
    import scipy.stats
    import string

    s_preds = []

    pred_dir = "./predictions/"
    if not os.path.exists(pred_dir): os.mkdir(pred_dir)
    # pred_file = open(pred_dir+expid+"txt", "w")

    for i, vid_idx in enumerate(set_vid_idxs):
        path = vid_paths[vid_idx]
        print path

        reader = bcolz.open(path, mode="r")
        max_frames = reader.shape[0]

        # max_frames = vid_meta[vid_idx]["max_frames"]
        # n_chunks = int(np.ceil((max_frames-cut_off) / float(stride)))
        s_pred = []
        j = 0
        while True:
            start_frame = j * stride
            end_frame = start_frame + cfg.n_frames
            if j != 0 and end_frame-cut_off >= max_frames: break
            j += 1

            fragment, start, end = vidprep.get_fragment(start_frame, reader, push_start=False)
            fraglen = len(fragment)
            if end_frame >= max_frames:
                in_vid = np.zeros((cfg.n_frames,) + cfg.im_shp, "float32")
                in_vid[:fraglen] = fragment
            else:
                in_vid = fragment

            in_vid.shape = (1,)+(cfg.n_frames,) + cfg.im_shp
            iter_result = iter_eval(in_vid)[0]
            if end_frame >= max_frames:
                cut_end = fraglen
            else:
                cut_end = cfg.n_frames-cut_off
            preds = np.argmax(iter_result[cut_off:cut_end], axis=1)
            # print start, end, preds
            s_pred.append(preds)
        del reader
        s_pred = np.hstack(s_pred)
        if len(s_pred) > 0:
            s_pred = np.hstack((np.repeat(s_pred[0],cut_off),s_pred)).astype("int32")
            if len(s_pred) != max_frames:
                last = scipy.stats.mode(s_pred[:-8])[0]
                s_pred = np.hstack( (s_pred, np.repeat(last, max_frames-len(s_pred))))
            # print path, max_frames, len(s_pred), s_pred
            # print classperframe.framewise_lbls[vid_idx]
            # print
        else:
            print
            print max_frames, len(s_pred)
            print

        s_preds.append(s_pred)

        # pred_file.write(string.join(path.split("/")[-2:], "/"))
        # begin = 1
        # prev = None
        # for f, p in enumerate(s_pred):
        #     if prev is None: prev = p
        #     elif prev != p:
        #         pred_file.write(" %i,%i:%i"%(begin, f, p+1))
        #         begin = f+1
        #         prev = p
        #     elif f == len(s_pred-1):
        #         pred_file.write(" %i,%i:%i" % (begin, f+1, p + 1))
        #
        # pred_file.write("\n")
        try:
            assert max_frames == len(s_pred)
        except: print "\n\t !!! ", max_frames, len(s_pred), "\n"

    cPickle.dump(s_preds, open(pred_dir+expid+".pkl", "wb"), protocol=cPickle.HIGHEST_PROTOCOL)



    # pred_file.close()

    sys.exit()

    chunk_gen = cfg.data_loader.chunk_generator(n_chunks="all", chunk_size=chunk_size, set=set_)

    jaccard = []
    preds = []

    for c, chunk in enumerate(chunk_gen):
        for key in data_shared:
            data_shared[key].set_value(chunk[key], borrow=cfg.borrow_shared_data)

        for b in range(cfg.batches_per_chunk):
            iter_result = iter_eval(b)[0]
            print iter_result.shape
            preds.append(iter_result)

        p = np.vstack(preds[-cfg.batches_per_chunk:]).reshape((-1, cfg.n_frames))
        jaccard.append(calc_ji(p, chunk["target"]))
        print c, jaccard[-1]

    print "mean JI =", np.mean(jaccard)


def calc_ji(preds, targets):
    p = preds[:,4:-4].flatten().astype("int32")
    p = np.eye(249)[p]
    targets = targets[:, 4:-4].flatten()
    targets = np.eye(249)[targets]
    intersection = np.sum(targets * p, axis=0)
    tarsum = np.sum(targets, axis=0)
    union = tarsum + np.sum(p, axis=0) - intersection
    ji = intersection / (union + 1e-9)
    return ji.sum() / np.count_nonzero(tarsum)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=str, default="metadata/default.pkl")
    parser.add_argument("--set", type=str, default=VALID)
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--bcolz", type=int, default=1)

    args, unknown = parser.parse_known_args()
    print args
    return args


if __name__ == '__main__':
    main()