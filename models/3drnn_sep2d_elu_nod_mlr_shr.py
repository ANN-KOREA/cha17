from collections import OrderedDict
import lasagne as nn
from lasagne.layers import DenseLayer, InputLayer, Conv2DLayer, MaxPool2DLayer, ReshapeLayer, DimshuffleLayer
from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer
from matplotlib.pyplot import axes

from scripts.load import LoaderCon as Loader
from scripts.prep import *
from utils.objectives import *
from utils import arr
from utils.extralayers import VarNormLayer
from utils.weight_norm import weight_norm
import paths

# INPUT DATA
############

batch_size = 16
batches_per_chunk = 1
chunk_size = batch_size*batches_per_chunk

im_shp = 128, 128
n_channels = 1
n_frames = 32
voc_size = 249 # n_classes

data_tags = OrderedDict(
    target=arr((n_frames,), "int32"),
    video=arr((n_frames,) + im_shp + (n_channels,), "float32"),
)
if n_channels == 1:
    data_tags["video"] = arr((n_frames,) + im_shp, "float32")

augm_params={
                "translation": [0, 32, 32],
                "rotation": [2, 0, 0],
                "shear": [2, 0, 0],
                "scale": [1, 1.2, 1.2],
                "reflection": [0, 0, 0]  # Bernoulli p
            }
print "augm_params", augm_params

data_loader = Loader(
    data_path=paths.CON_PREP2,
    preprocessors=[
        LabelsCon(),
        VideoLoadPrep(n_frames=n_frames, rand_middle_frame=True, rgb=True, depth=False,
                      use_bcolz=True, tolerance=0, rgbbias=-127),
        Augment3D(output_shape=(n_frames,)+im_shp, output_scale=(1,1,1), mode="constant",  dbias=0,
                  interp_order=1, augm_params=augm_params),
        ClassPerFrame(n_frames=n_frames),
    ],
    inputs = data_tags
)

# TRAINING
############

learning_rate = 1e-4 * batch_size
learning_rate_decay = 3e-5

validate_every_n_samples = 20*1024
validate_every_n_chunks = int(np.ceil(validate_every_n_samples/float(chunk_size)))

print_every_n_samples = 512
print_every_n_chunks = int(np.ceil(print_every_n_samples/float(chunk_size)))

toprint = OrderedDict()
n_valid_samples = 2*1024
n_valid_chunks = int(np.ceil(n_valid_samples/float(chunk_size)))
borrow_shared_data = False
n_updates = int(1e10)

# MODEL
############

# normalize = weight_norm # nn.layers.batch_norm # lambda l: l
normalize = nn.layers.batch_norm # lambda l: l
# normalize = lambda l: l
# init = nn.init.Orthogonal("relu")
init = nn.init.HeNormal()
nonlinearity = nn.nonlinearities.elu

def build_model():
    l_src_in = InputLayer(shape=(None,) + data_tags["video"].shape)
    l_tar = InputLayer(shape=(None,) + data_tags["target"].shape, input_var=T.imatrix("target"))

    l = l_src_in
    if n_channels == 1: l = nn.layers.DimshuffleLayer(l, (0,"x",1,2,3))
    else: l = nn.layers.DimshuffleLayer(l, (0,4,1,2,3))
    # b, c, t, h, w

    # l = VarNormLayer(l, axis=(3, 4))
    l = nn.layers.ExpressionLayer(l, lambda x: x/255.)

    # shp = l.output_shape
    # l = nn.layers.ReshapeLayer(l, (-1,)+shp[2:])

    n = 16
    l = conv3d(l, num_filters=n, filter_size=7, stride=(1,2,2))
    # l = maxpool2d(l, pool_size=3, stride=2)

    n += 16
    l = res(l, num_filters=n, stride=(1,2,2))
    l = res(l, num_filters=n)
    l = res(l, num_filters=n)

    n += 16
    l = res(l, num_filters=n, stride=(1,2,2))
    l = res(l, num_filters=n)
    l = res(l, num_filters=n)

    n += 16
    l = res(l, num_filters=n, stride=(1,2,2))
    l = res(l, num_filters=n)
    l = res(l, num_filters=n)

    n += 16
    l = res(l, num_filters=n, stride=(1,2,2))
    l = res(l, num_filters=n)
    l = res(l, num_filters=n)

    # shp = l.output_shape
    # l = nn.layers.ReshapeLayer(l, (-1,n_frames) + shp[1:])

    l = nn.layers.ExpressionLayer(l, lambda x: T.mean(x.flatten(4), axis=3), output_shape=lambda x: x[:3])
    # b, c, t

    l = nn.layers.DimshuffleLayer(l, (0, 2, 1)) # b, t, c

    l = drop(l)

    n += 16
    l_forw = lstm(l, num_units=n)
    l_back = lstm(l, num_units=n, backwards=True)
    l = nn.layers.ElemwiseSumLayer([l_forw, l_back])
    # l = l_forw

    l = drop(l)


    shp = l.output_shape
    l = nn.layers.ReshapeLayer(l, (-1,) + shp[2:])

    l_out = softmaxlayer(l, num_units=voc_size)

    return OrderedDict(
        input=OrderedDict(
            target=l_tar,
            video=l_src_in
        ),
        output=l_out
    )

# OBJECTIVE
############

def build_objectives(model, deterministic):
    preds = nn.layers.get_output(model["output"], deterministic=deterministic)
    preds = T.clip(preds, 1e-9, 1)
    targets = model["input"]["target"].input_var
    targets = T.flatten(targets)

    loss = nn.objectives.categorical_crossentropy(preds, targets)
    loss = loss.mean()

    if not deterministic: return loss
    else:
        return OrderedDict(
            loss=loss,
            _preds=T.argmax(preds, axis=1)
        )

# UPDATES
############

def build_updates(grads, params, learning_rate):
    # toprint["grad_norm"] = T.sqrt(T.sum([(g**2).sum() for g in grads])+1e-9)
    # grads = nn.updates.total_norm_constraint(grads, max_norm=4)
    return nn.updates.adam(grads, params, learning_rate)
    # return nn.updates.nesterov_momentum(grads, params, learning_rate)
    # return nn.updates.rmsprop(grads, params, learning_rate)

# PREPARE MODEL
###############

def preparation(model, batch_idx, givens, data_shared):
    if normalize == weight_norm:
        init_weight_norm = 64 # number of samples per layer to init the weights with

        from scripts.weight_norm_init import init_weights
        init_weights(model, batch_idx, givens, data_shared, init_weight_norm, chunk_size, batches_per_chunk, data_loader)

# CUSTOM EVALUATION
#####################
# def evaluate(chunk_outputs, chunk_data, expid, set_): pass

# BUILDING BLOCKS
#################

_conv3d = partial(Conv3DDNNLayer, stride=1, filter_size=3, pad="same", W=init, nonlinearity=nonlinearity)

_conv2d = partial(Conv2DLayer, stride=1, filter_size=3, pad="same", W=init, nonlinearity=nonlinearity)

_dense = partial(DenseLayer, W=init, nonlinearity=nonlinearity)

maxpool3d = partial(MaxPool3DDNNLayer, pool_size=2)
maxpool2d = partial(MaxPool2DLayer, pool_size=2)

conv3d = lambda l, **kwargs: normalize(_conv3d(l, **kwargs))
conv2d = lambda l, **kwargs: normalize(_conv2d(l, **kwargs))
dense = lambda l, **kwargs: normalize(_dense(l, **kwargs))
softmaxlayer = partial(DenseLayer, W=nn.init.Constant(0.), nonlinearity=nn.nonlinearities.softmax)

drop = nn.layers.DropoutLayer

def residual(l_in, num_filters, normalization=None, filter_size=3, stride=1, num_layers=2,
             nonlinearity=nn.nonlinearities.rectify, W=nn.init.GlorotUniform(), sep=False):
    if not isinstance(stride, tuple): stride = (stride,)*3
    l = l_in
    c3d = partial(conv3d, num_filters=num_filters, filter_size=filter_size, stride=stride, pad='same', nonlinearity=nonlinearity, W=W)

    if (num_filters != l.output_shape[1]) or (stride != 1):
        l_in = c3d(l_in, filter_size=1, pad=0, nonlinearity=None, b=None)

    for _ in range(num_layers):
        if sep:
            l = _conv3d(l, stride=(1,stride[1],stride[2]), filter_size=(1,filter_size,filter_size),
                        num_filters=num_filters, pad='same', nonlinearity=None, W=W, b=None)
            l = c3d(l, stride=(stride[0],1,1), filter_size=(filter_size,1,1))
        else:
            l = c3d(l, stride=stride)
        stride = (1,)*3 # only first one

    l.nonlinearity = nn.nonlinearities.identity
    l = nn.layers.ElemwiseSumLayer([l, l_in], coeffs=0.5)
    l = nn.layers.NonlinearityLayer(l, nonlinearity=nonlinearity)

    return l


res = partial(residual, normalization=normalize, W=init, nonlinearity=nonlinearity, sep=True)

lstm = partial(nn.layers.LSTMLayer, unroll_scan=True)

rnn = partial(nn.layers.RecurrentLayer, unroll_scan=True)