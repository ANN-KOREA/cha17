import theano
import numpy as np
import lasagne as nn

from utils import TRAIN
from utils.weight_norm import WeightNormLayer, weight_norm, debuglog


DEBUG = False


def init_weights(model, batch_idx, givens, data_shared, init_weight_norm, chunk_size, batches_per_chunk, data_loader):
    print "Initializing weights with %i samples..." % init_weight_norm

    for layer_idx in range(WeightNormLayer.layer_count):
        print "\tWN layer", layer_idx
        n_chunks = int(np.ceil(init_weight_norm / float(chunk_size) ))
        chunk_gen = data_loader.chunk_generator(n_chunks=n_chunks, chunk_size=chunk_size, set=TRAIN)

        nn.layers.helper.get_output(model["output"], None, init=layer_idx, debug=DEBUG)
        all_layers = nn.layers.get_all_layers(model["output"])
        init_updates = [u for l in all_layers for u in getattr(l, 'init_updates', [])]
        apply_updates = [u for l in all_layers for u in getattr(l, 'apply_updates', [])]
        init_wn_fun = theano.function([batch_idx], outputs=debuglog.values(),
                                      givens=givens, updates=init_updates, on_unused_input='ignore')
        apply_wn_fun = theano.function([batch_idx], outputs=None, givens=givens, updates=apply_updates, on_unused_input='ignore')

        for c, chunk in enumerate(chunk_gen):
            for key in data_shared: data_shared[key].set_value(chunk[key])
            for b in range(batches_per_chunk):
                debugres = init_wn_fun(b)
                for i, res in enumerate(debugres): print debuglog.keys()[i], res

        apply_wn_fun(0)
        del init_wn_fun
        del apply_wn_fun
        del chunk_gen