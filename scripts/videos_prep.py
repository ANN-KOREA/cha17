import os
import sys
import bcolz
import cv2
import numpy as np
from glob import glob
from moviepy.editor import VideoFileClip

sys.path.append(".")
import paths
from utils import rgb2gray
from utils.plt import show_animate

bcolz.set_nthreads(4)
def main():
    dst_dir = paths.CON_PREP2
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
        print

        v_d = get_vid(path)
        v_d = prep(v_d)

        v_rgb = get_vid(path.replace(".K.", ".M."), rgb=True)
        v_rgb = preprgb(v_rgb)

        if v_d.shape != v_rgb.shape: raise Exception(""+str(v_d.shape)+" "+str(v_rgb.shape))

        v = np.empty(v_rgb.shape+(2,), "uint8")

        v[...,0] = v_rgb
        v[...,1] = v_d

        # v = v_rgb
        # if v.shape[-1] == 1:
        #     v = np.repeat(v, 3, -1)
        # elif v.shape[-1] == 2:
        #     v = np.concatenate((v, np.zeros_like(v[..., 0:1])), -1)
        # show_animate(v / 255., fps=12.5, figsize=(7, 7), interpolation="nearest", normalize=False)


        save_bcolz(v, dst)
    print
    print

def preprgb(v):
    # v = 255 - v
    # v = v.astype("float32") - np.mean(v)
    # v = (v / v.max()) * 255.

    new_v = np.empty((v.shape[0],128,128), "uint8")
    prev = np.zeros(v.shape[1:], "float32")
    for i, f in enumerate(v):
        if i == 0:
            prev = f
            f = np.zeros(v.shape[1:], "float32")
        else:
            diff = ((f.astype("float32")-prev)+255.)/2.
            prev = f
            f = diff
            f = np.clip(f,0,255).astype("uint8")
        f = cv2.resize(f, (128, 128), interpolation=cv2.INTER_AREA)
        new_v[i] = f.astype("uint8")

    return new_v


def prep(v):
    v = 255 - v
    for i, f in enumerate(v):
        v[i] = cv2.medianBlur(f, 15)

    v = v.astype("float32") - np.mean(v)
    v = (v / v.max()) * 255.

    new_v = np.empty((v.shape[0],128,128), "uint8")
    for i, f in enumerate(v):
        f = np.clip(f,0,255).astype("uint8")
        f = cv2.resize(f, (128, 128), interpolation=cv2.INTER_AREA)
        new_v[i] = f.astype("uint8")

    return new_v


def get_vid(path, rgb=False):
    clip = VideoFileClip(path)
    # n_frames = clip.fps * clip.duration
    # print n_frames
    vid = []

    if rgb:
        for i, frame in enumerate(clip.iter_frames()):
            vid.append(cv2.cvtColor(frame,cv2.cv.CV_RGB2GRAY))
    else:
        for i, frame in enumerate(clip.iter_frames()):
            vid.append(frame[...,0])
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

def inpaint(img, threshold):
    mask = img.copy()
    mask[mask<threshold] = 0
    cv2.inpaint(img, mask, 9, cv2.INPAINT_TELEA, img)
    img[img>threshold] = 0
    return img



if __name__ == '__main__':
    main()

