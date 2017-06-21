import numpy as np
from copy import deepcopy
import matplotlib
import sys
import string
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import time

sys.setrecursionlimit(10000)


stopwatch = lambda t0: (time()-t0)*1000


def detect_nans(loss, all_params, data_shared):
    if not np.isnan(loss): return
    print "NaN Detected."
    for p in all_params:
        if not np.isfinite(p.get_value()).all(): print "Nan detected in", p.name
    for tag, v in data_shared.iteritems():
        if not np.isfinite(v.get_value()).all(): print "Nan detected in loaded data: %s"%tag
    raise Exception("NaN's")


def plot_progress(train, val, path, max_points=100):
    values = [deepcopy(v) for v in (train, val)]
    l = min(max_points, len(train))
    x = np.linspace(0, len(train), l)

    # if len(train) > max_points:
    for i, v in enumerate(values):
        values[i] = np.interp(
            np.linspace(0, len(v) - 1, l),
            np.linspace(0, len(v) - 1, len(v)),
            v)

    plt.ylim([min(np.min(values[0]), np.min(values[1])),max(np.max(values[0]), np.max(values[1]))])
    plt.plot(x, values[0], 'b', x, values[1], 'r')
    try: plt.savefig(path,bbox_inches='tight')
    except: print "Failed saving figure"
    plt.close()


class TrainLogger(object):
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "a")
        self.log_path = log_path

    def write(self, message):
        try:
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        except: pass

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        self.log.flush()

    def print_to_log(self): sys.stdout = self

    @staticmethod
    def only_print_to_console(): sys.stdout = sys.__stdout__

    def __del__(self):
        self.log.close()


def print_model(all_layers, all_params, exclude=None):
    if exclude is None: exclude = {}
    print string.ljust("  layer output shapes:", 26),
    print string.ljust("#params:", 10),
    print string.ljust("#data:", 10),
    print "output shape:"

    def comma_seperator(v):
        return '{:,.0f}'.format(v)

    for layer in all_layers:
        if layer.__class__.__name__ in exclude: continue
        name = string.ljust(layer.__class__.__name__, 22)
        num_param = sum([np.prod(p.get_value().shape) for p in layer.get_params()])
        num_param = string.ljust(comma_seperator(num_param), 10)
        num_size = string.ljust(comma_seperator(np.prod(layer.output_shape[1:])), 10)
        print "    %s %s %s %s" % (name, num_param, num_size, layer.output_shape)

    num_params = sum([np.prod(p.get_value().shape) for p in all_params])
    print "  number of parameters:", comma_seperator(num_params)
