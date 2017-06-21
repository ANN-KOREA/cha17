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
    cfg_name = args.config.split("/")[-1][:-3]
    expid = "%s-%s-%s" % (cfg_name, platform.node(), strftime("%Y%m%d-%H%M%S", localtime()))
    cfg = importlib.import_module("models.%s" % cfg_name)
    is_resume = args.resume != ""
    if is_resume:
        meta = cPickle.load(open(args.resume, "rb"))
        cfg_name = meta["configuration"]
        expid = meta["experiment_id"]
        params = meta["parameters"]
    train_loss = meta["train_loss"] if is_resume else []
    valid_loss = meta["valid_loss"] if is_resume else []
    start_epoch = meta["epoch"] if is_resume else -1
    if is_resume: resume_learning_rate(cfg, start_epoch)

    if not os.path.exists("metadata/"): os.mkdir("metadata")

    log = TrainLogger("metadata/%s.log"%expid)
    log.print_to_log()

    print "Model:", cfg_name,  "expid:", expid
    print "batch_size", cfg.batch_size, "batches_per_chunk", cfg.batches_per_chunk, \
        "learning_rate", cfg.learning_rate, "validate_every_n_chunks", cfg.validate_every_n_chunks, \
        "n_valid_chunks", cfg.n_valid_chunks


    cfg.data_loader.start(n_jobs=args.jobs)

###############################################################################################################
# BUILD
###############################################################################################################

    print "Building model ..."
    model = cfg.build_model()
    if is_resume: nn.layers.helper.set_all_param_values(model["output"], params)
    if hasattr(cfg,"is_pretrained") and cfg.is_pretrained: cfg.set_pretrained_params(model)
    all_layers = nn.layers.get_all_layers(model["output"])
    all_params = nn.layers.get_all_params(model["output"], trainable=True)
    print_model(all_layers, all_params)#, {"WeightNormLayer", "LayerNormLayer", "BatchNormLayer", "NonlinearityLayer"})

    print "Building objectives ..."
    loss = cfg.build_objectives(model, deterministic=False)
    eval_outputs = cfg.build_objectives(model, deterministic=True)  # ordered dict

    print "Building updates ..."
    learning_rate = theano.shared(np.float32(cfg.learning_rate))
    all_grads = theano.grad(loss, all_params, disconnected_inputs='warn')
    updates = cfg.build_updates(all_grads, all_params, learning_rate)


    print "Compiling training function"
    train_outputs = [loss]
    train_outputs += cfg.toprint.values() #OrderedDict!

    batch_idx = T.iscalar('idx')
    data_shared = OrderedDict({tag: nn.utils.shared_empty(len(arr.shape)+1, arr.dtype)
                               for (tag, arr) in cfg.data_tags.items()})
    givens = OrderedDict()
    for (tag, l) in model["input"].items():
            givens[l.input_var] = data_shared[tag][batch_idx*cfg.batch_size : (batch_idx+1)*cfg.batch_size]

    iter_train = theano.function([batch_idx], train_outputs, updates=updates, givens=givens)

    print "Compiling validation function"
    iter_valid = theano.function([batch_idx], eval_outputs.values(), givens=givens)

###############################################################################################################
# PREPARE
###############################################################################################################

    if hasattr(cfg, "preparation") and not is_resume:
        print "Preparing model (weight init etc)..."
        t_init = time()
        cfg.preparation(model, batch_idx, givens, data_shared)
        print "  took %.3f seconds" % (time() - t_init,)

###############################################################################################################
# TRAIN
###############################################################################################################

    print "Training ..."

    if hasattr(args, "bcolz"):
        import bcolz
        bcolz.set_nthreads(args.bcolz)
    train_time, total_time = 0, time()
    start_time = time()
    chunk_size = cfg.batch_size * cfg.batches_per_chunk
    n_epochs = int(cfg.n_updates / float(cfg.validate_every_n_chunks*cfg.batches_per_chunk))
    chunk_loss = []
    valid_jac = []
    for epoch in xrange(start_epoch + 1, n_epochs):

        # EVALUATION

        if epoch != 0:
        # if True:
            for loss_list, set_ in ((valid_loss, VALID), (train_loss, TRAIN)):
                print "Evaluating %s set..."%set_
                valid_time = time()
                chunk_gen = cfg.data_loader.chunk_generator(n_chunks=cfg.n_valid_chunks, chunk_size=chunk_size, set=set_)

                chunk_res = OrderedDict({tag: [] for tag in eval_outputs.keys()})
                chunk_res["JI"] = []

                for c, chunk in enumerate(chunk_gen):
                    for key in data_shared:
                        data_shared[key].set_value(chunk[key], borrow=cfg.borrow_shared_data)

                    for b in range(cfg.batches_per_chunk):
                        iter_valid_result = iter_valid(b)
                        for i, tag in enumerate(eval_outputs.keys()):
                            chunk_res[tag].append(iter_valid_result[i])

                    preds = np.vstack(chunk_res["_preds"][-cfg.batches_per_chunk:]).reshape((-1,cfg.n_frames))[:,4:-4].flatten().astype("int32")
                    preds = np.eye(249)[preds]
                    targets = chunk["target"][:,4:-4].flatten()
                    targets = np.eye(249)[targets]

                    intersection = np.sum(targets * preds, axis=0)
                    tarsum = np.sum(targets, axis=0)
                    union = tarsum + np.sum(preds, axis=0) - intersection
                    ji = intersection / (union + 1e-9)
                    ji = ji.sum() / np.count_nonzero(tarsum)
                    # print preds.shape, targets.shape, preds, targets
                    # ji = jaccard_similarity_score(targets, preds)
                    # print preds.shape, preds.dtype, targets.shape, targets.dtype, ji, np.count_nonzero(tarsum)
                    chunk_res["JI"].append(ji)

                    if hasattr(cfg, "evaluate"): cfg.evaluate(chunk_res, chunk, expid, set_)

                loss_list.append(np.mean(chunk_res["loss"]))
                if set_ == VALID: valid_jac.append(np.mean(chunk_res["JI"]))

                toprint = "best=%.3f"% (np.min(loss_list) if set_ == TRAIN else np.max(valid_jac))
                for tag, res in chunk_res.items():
                    if tag.startswith("_"): continue
                    toprint += " %s=%.3f" % (tag, np.mean(res))

                print toprint

                # chunk_res["_preds"]

        # SAVING PARAMS

        if epoch != 0 and valid_jac[-1] == np.max(valid_jac):
        # if valid_jac[-1] == np.max(valid_jac):
            try:
                metadata_tmp_path = "/var/tmp/%s.pkl" % expid
                metadata_target_path = "metadata/%s.pkl" % expid
                print "Saving in", metadata_target_path

                with open(metadata_tmp_path, 'w') as f:
                    cPickle.dump({
                        'configuration': cfg_name,
                        'experiment_id': expid,
                        'train_loss': train_loss,
                        'valid_loss': valid_loss,
                        'parameters': nn.layers.get_all_param_values(model["output"]),
                        'epoch': epoch,
                    }, f, cPickle.HIGHEST_PROTOCOL)

                try:
                    shutil.move(metadata_tmp_path, metadata_target_path)
                except Exception as e:
                    print e
            except:
                print "saving failed"

        if epoch != 0:
            plot_progress(train_loss, valid_loss, "metadata/%s--plot.pdf"%expid)
            print "Evaluation time:%.3fs" % (time()-valid_time)

        # TRAINING

        chunk_gen = cfg.data_loader.chunk_generator(n_chunks=cfg.validate_every_n_chunks,
                                        chunk_size=chunk_size,
                                        set=TRAIN)

        for c, chunk in enumerate(chunk_gen):
            for key in data_shared:
                data_shared[key].set_value(chunk[key], borrow=cfg.borrow_shared_data)

            total_n_chunks = epoch * cfg.validate_every_n_chunks + c
            if total_n_chunks % cfg.print_every_n_chunks == 0:
                sys.stdout.write("\r" + " " * 100 + "\r"); sys.stdout.flush()
                log.print_to_log()
                print "Chunk %i updates %i samples %i lr %.2e time %s" % \
                      (total_n_chunks,
                       total_n_chunks * cfg.batches_per_chunk,
                       total_n_chunks * chunk_size,
                       learning_rate.get_value(),
                       secs_to_str(time() - start_time))
                total_time = stopwatch(total_time)
                print "Time / sample = %.3fms (%.3fms + %.3fms overhead)" % \
                      (total_time / (cfg.print_every_n_chunks * chunk_size),
                       train_time / (cfg.print_every_n_chunks * chunk_size),
                       (total_time - train_time) / (cfg.print_every_n_chunks * chunk_size))
                train_time, total_time = 0, time()
                val_loss = valid_jac[-1] if len(valid_jac) > 0 else np.inf
                min_val_loss = np.max(valid_jac) if len(valid_jac) > 0 else np.inf
                print "Train loss = %.3f,  Valid loss = %.3f (best: %.3f)\n" % \
                      (np.mean(chunk_loss), val_loss, min_val_loss)
                chunk_loss = []

            log.only_print_to_console()

            batch_loss = []
            for b in range(cfg.batches_per_chunk):
                t0 = time()
                iter_train_result = iter_train(b)
                train_time += stopwatch(t0)

                batch_loss.append(iter_train_result[0])

                # learning decay
                new_lr = cfg.learning_rate * calculate_lr_decay(cfg, epoch, c, b)
                learning_rate.set_value(np.float32(new_lr))

                toprint = "\r" + " " * 100 + "\rl=%.4f" % (batch_loss[-1],)
                for i, res in enumerate(iter_train_result[1:]):
                    toprint += " %s=%.3f" % (cfg.toprint.keys()[i], res)
                sys.stdout.write(toprint); sys.stdout.flush()

            chunk_loss.append(np.mean(batch_loss))

            detect_nans(chunk_loss[-1], all_params, data_shared)

        # end of training per epoch

        sys.stdout.write("\r" + " " * 100 + "\r"); sys.stdout.flush()
        log.print_to_log()

    # end of training


def resume_learning_rate(cfg, start_epoch):
    cfg.learning_rate *= calculate_lr_decay(cfg, start_epoch, 0, 0)
    print
    print
    print "LEARNING RATE", cfg.learning_rate
    print
    print


def calculate_lr_decay(cfg, epoch, chunk_idx, batch_idx):
     return (1. - cfg.learning_rate_decay) ** (
         epoch * cfg.validate_every_n_chunks * cfg.batches_per_chunk +
         chunk_idx * cfg.batches_per_chunk +
         batch_idx)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="models/default.py")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--bcolz", type=int, default=1)

    args, unknown = parser.parse_known_args()
    print args
    return args


if __name__ == '__main__':
    main()