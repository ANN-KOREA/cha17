import lasagne as nn
import theano.tensor as T
import theano
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lasagne.random import get_rng

class VarNormLayer(nn.layers.Layer):
    """
    Variance normalization across given dimensions.
    """
    def __init__(self, input_layer, axis=-1, ignore0=False, epsilon=1e-9):
        super(VarNormLayer, self).__init__(input_layer)
        self.axis = axis
        self.epsilon = epsilon
        self.ignore0 = ignore0

    def get_output_for(self, input, *args, **kwargs):
        return (input-T.mean(input, axis=self.axis, keepdims=True, dtype=theano.config.floatX)) / \
            T.sqrt(T.var(input, axis=self.axis, keepdims=True)+self.epsilon)



class ModDropLayer(nn.layers.Layer):
    def __init__(self, incoming, p=0.5, **kwargs):
        super(ModDropLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.p = p

    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        deterministic : bool
            If true dropout and scaling is disabled, see notes
        """
        if deterministic or self.p == 0:
            return input
        else:
            zero = T.constant(0)
            one = T.constant(1)

            retain_prob = one - self.p
            input /= retain_prob

            # use nonsymbolic shape for dropout mask if possible
            input_shape = self.input_shape
            if any(s is None for s in input_shape):
                input_shape = input.shape

            b = self._srng.binomial((input_shape[0],), p=retain_prob, dtype=input.dtype).dimshuffle((0,"x", "x", "x", "x"))

            ones = T.ones((input_shape[0], 2))
            first = (ones*T.stack((one, zero))).dimshuffle((0,1,"x", "x", "x"))
            second = (ones*T.stack((zero, one))).dimshuffle((0,1,"x", "x", "x"))

            mask = b*first + (one-b)*second

            return input * mask