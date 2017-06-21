import os
import sys
import bcolz
import numpy as np
from glob import glob
from moviepy.editor import VideoFileClip

sys.path.append(".")
import paths
from utils import rgb2gray
from utils.plt import show_animate

bcolz.set_nthreads(4)
def main():
    dst_dir = paths.CON_PREP
    if not os.path.exists(dst_dir): os.mkdir(dst_dir)

    dump(dst_dir+"train/", glob(paths.CON_VID_TRAIN))
    dump(dst_dir+"valid/", glob(paths.CON_VID_VALID))
    dump(dst_dir+"test/", glob(paths.CON_VID_TEST))

def dump(dst_dir, vid_paths):
    if not os.path.exists(dst_dir): os.mkdir(dst_dir)

    vid_paths.sort()
    print dst_dir, len(vid_paths)
    vid_paths = [p for p in vid_paths if p[-5]=="K"]
    # print len(vid_paths)

    for i, path in enumerate(vid_paths):
        class_dir = dst_dir + path.split("/")[-2] + "/"
        if not os.path.exists(class_dir): os.mkdir(class_dir)
        dst = class_dir + os.path.basename(path).split(".")[0]
        # sys.wrt "\r", i + 1, "/", len(vid_paths), dst
        toprint = "\r" + " " * 100 + "\r%.3f%% %s" % (100.*(i + 1)/float(len(vid_paths)), dst)
        sys.stdout.write(toprint); sys.stdout.flush()

        vid_d = get_vid(path)
        vid_rgb = get_vid(path.replace(".K.", ".M."))

        if vid_d.shape != vid_rgb.shape: raise Exception(""+str(vid_d.shape)+" "+str(vid_rgb.shape))

        vid = np.empty(vid_rgb.shape[:-1]+(2,), "uint8")
        # print vid.shape

        for j, frame in enumerate(vid_rgb):
            vid[j, ..., 0] = rgb2gray(frame)
            vid[j, ..., 1] = vid_d[j,...,0]

        # v = vid
        # if v.shape[-1] == 1:
        #     v = np.repeat(v, 3, -1)
        # elif v.shape[-1] == 2:
        #     v = np.concatenate((v, np.zeros_like(v[..., 0:1])), -1)
        # show_animate(v / 255., fps=12.5, figsize=(7, 7), interpolation="nearest", normalize=False)

        save_bcolz(vid, dst)
    print
    print


def get_vid(path):
    clip = VideoFileClip(path)
    # n_frames = clip.fps * clip.duration
    # print n_frames
    vid = []
    for i, frame in enumerate(clip.iter_frames()):
        vid.append(frame)
    v = np.array(vid, dtype="uint8")
    return v


def save_bcolz(data, rootdir):
    data_bcolz = bcolz.carray(array=data,
                              # chunklen=data.shape[0],
                              dtype="uint8",
                              cparams=bcolz.cparams(clevel=1, cname="zlib"),  # lz4hc zlib blosc
                              rootdir=rootdir,
                              mode="w")
    data_bcolz.flush()


if __name__ == '__main__':
    main()

