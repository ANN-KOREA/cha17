import numpy as np
import os
from time import sleep
from functools import partial

from signal import signal, SIGINT, SIGKILL



def show_animate(data, fps=25, normalize=False, figsize=None, interpolation='nearest'):
    """interpolation: 'nearest', 'bilinear', 'bicubic', 'spline16',
           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'"""
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib import cm

    def _terminate(sig=None, frame=None, pgrp=0):
        os.killpg(pgrp, sig)
        sleep(1)
        os.killpg(pgrp, SIGKILL)

    signal(SIGINT, partial(_terminate, pgrp=os.getpgrp()))

    if data.dtype != np.float32:
        data = data.astype(np.float32)

    if normalize:
        mini = data.min()
        data = (data-mini)/(data.max()-mini)

    if data.ndim == 4 and data.shape[-1] == 1: data = data[...,0]

    if data.ndim == 4 and data.shape[-1] == 2:
        data = np.concatenate((data, np.zeros_like(data[..., 0:1])), -1)

    if data.ndim == 3:
        data = np.concatenate((data[...,None], data[...,None], data[...,None]), axis=-1)

    def get_data_step(step): return data[step]

    if normalize: norm = cm.colors.Normalize()
    else: norm = cm.colors.NoNorm()

    fig = plt.figure(figsize=figsize)
    im = fig.gca().imshow(get_data_step(0), interpolation=interpolation, norm=norm)

    # initialization function: plot the background of each frame
    def init():
        im.set_data(get_data_step(0))
        return im,

    # animation function.  This is called sequentially
    def animate(i):
        im.set_data(get_data_step(i))
        return im,

    interval = 1000./float(fps)
    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=data.shape[0], interval=interval, blit=True)
    plt.show()


