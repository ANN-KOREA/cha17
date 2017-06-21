import lasagne as nn


def residual_block(input_layer, normalization=None, bottleneck=False, stride=1,
                 nonlinearity=nn.nonlinearities.rectify, **kwargs):

        l_in = input_layer

        l = nn.layers.Conv2DLayer(l, filter_size=3, nonlinearity=nonlinearity, **kwargs)
        if normalization is not None: l = normalization(l)

        l = nn.layers.Conv2DLayer(l, filter_size=3, nonlinearity=None, **kwargs)
        if normalization is not None: l = normalization(l)

        l = nn.layers.ElemwiseSumLayer([l, l_in])

        l = nn.layers.NonlinearityLayer(l, nonlinearity=nonlinearity)

