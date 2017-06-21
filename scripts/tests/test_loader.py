import sys
from time import time
import numpy as np

sys.path.append(".")
from utils import arr, TRAIN, VALID, invert_dict
import paths
from utils.plt import show_animate

from scripts.prep import *
from scripts.load import LoaderCon

n_frames = 32
n_channels = 1
output_shape = n_frames, 128, 128 #240, 320

l = LoaderCon(
    data_path=paths.CON_PREP2,
    inputs={
        # "target": arr((1,), "int32"),
        "target": arr((n_frames,), "int32"),
        # "video": arr(output_shape+(n_channels,), "float32"),
        "video": arr(output_shape, "float32"),
        "path": arr((), "S128"),
    },
    preprocessors=[
        LabelsCon(),
        VideoLoadPrep(n_frames=n_frames, rand_middle_frame=True, rgb=True, depth=False,
                      use_bcolz=True, tolerance=0, rgbbias=-127.5),
        Augment3D(output_shape=output_shape, output_scale=(1,1,1), mode="constant", interp_order=1, dbias=0,
            augm_params={
                "translation": [0, 0, 0],
                "rotation": [0, 0, 0],
                "shear": [0, 0, 0],
                "scale": [1, 2, 2],
                "reflection": [0, 0, .5]  # Bernoulli p
            }
        ),
        ClassPerFrame(n_frames=n_frames)
    ]
)
l.start(n_jobs=0)

chunk_gen = l.chunk_generator(n_chunks=10, chunk_size=2, set=TRAIN)

t0 = time()
for i, chunk in enumerate(chunk_gen):#pass
    # print chunk["target"][0]
    t = time()-t0
    print "time", i, "%.3fms"%(t*1000.,)
    # print chunk["video"].shape
    vid_chunk = chunk["video"]
    # print vid_chunk.min(), vid_chunk.max(), vid_chunk.mean()
    for j, v in enumerate(vid_chunk):
        v += 127.5
        print "target", chunk["target"][j], chunk["path"][j]
        if v.shape[-1] == 1: v =  np.repeat(v, 3, -1)
        elif v.shape[-1] == 2:
            v = np.concatenate((v[...,0:1], v[...,1:2]), 2)
            v = np.repeat(v, 3, -1)
        else: v = np.repeat(v[...,None], 3, -1)
        # print v.min(), v.max()
        show_animate(np.clip(v/255.,0,1), fps=12.5, figsize=(10,5), interpolation="nearest", normalize=False)
    t0 = time()

print "\nvalid\n"

chunk_gen = l.chunk_generator(n_chunks=5, chunk_size=2, set=VALID)
for i, chunk in enumerate(chunk_gen):
    print chunk["target"][0]
    # vid_chunk = chunk["video"]
    # for v in vid_chunk: show_animate(v/255., fps=25)

print

chunk_gen = l.chunk_generator(n_chunks=10, chunk_size=2, set=TRAIN)

for i, chunk in enumerate(chunk_gen): pass

print

chunk_gen = l.chunk_generator(n_chunks=10, chunk_size=2, set=TRAIN)

for i, chunk in enumerate(chunk_gen): pass