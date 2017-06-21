import random
import bcolz
from functools import partial
import numpy.random as npr
import sys
import shutil
import os
import cPickle
from moviepy.editor import VideoFileClip

from utils.vidreader import extract_video_fragment
from utils import TRAIN, VALID

bcolz.set_nthreads(1)


class VideoLoadPrep(object):
    def __init__(self, n_frames=16, rand_middle_frame=False, rgb=True, depth=True, load_in_mem=False, tolerance=0, offset=0, use_bcolz=True, rgbbias=0):
        self.n_frames = n_frames
        self.rand_middle_frame = rand_middle_frame
        self.offset = offset
        self.tolerance = tolerance
        self.rgb = rgb
        self.depth = depth
        self.use_bcolz = use_bcolz
        self.load_in_mem = load_in_mem
        self.rgbbias = rgbbias

        #DEBUG
        # self.extract_fragment = lambda x,y: 0


    def prepare(self, loader):
        if self.use_bcolz: return

        new_paths = []
        for i, p in enumerate(loader.vid_paths):
            path = p.replace("preproc/con_bcolz/", "") + ".K.avi"
            new_paths.append(path)
        self.vid_paths = new_paths

        #load dataset into memory if not in memory yet
        if self.load_in_mem:
            target_dir = "/dev/shm/lpigou_cha17/"
            if not os.path.exists(target_dir): os.mkdir(target_dir)
            for i, path in enumerate(self.vid_paths):
                set_ = TRAIN if TRAIN in path else VALID
                target = target_dir+set_+os.path.basename(path)
                if not os.path.exists(target):
                    print "Copying data to", target
                    shutil.copyfile(path, target)
                self.vid_paths[i] = target

        target = "./data/vidmeta.pkl"
        if os.path.exists(target):
            self.meta = cPickle.load(open(target, "rb"))
            return

        self.meta = []
        for i, p in enumerate(loader.vid_paths):
            print i+1, "/", len(loader.vid_paths)
            path = p.replace("preproc/con_bcolz/", "") + ".K.avi"
            clip = VideoFileClip(path)
            fps = clip.fps
            max_frames = int(round(clip.duration * fps))
            self.meta.append({"fps": fps, "max_frames":max_frames})

        cPickle.dump(self.meta, open(target, "wb"), protocol=cPickle.HIGHEST_PROTOCOL)

    def process(self, sample):
        #DEBUG
        # if sample["set"] == TRAIN: random.seed(546548)

        path = sample["path"]
        begin = sample["begin"]
        end = sample["end"]

        # if npr.RandomState(sample["vid_id"]).rand() < (sample["smpl_idx"]/15.):
        #     path = path.replace("/data/lpigou", "/mnt/ssd")

        if self.use_bcolz:

            reader = bcolz.open(path, mode="r")

            if self.rand_middle_frame: frame = random.randint(begin-self.tolerance, end+self.tolerance)
            else: frame = (end + begin) // 2+random.randint(-self.tolerance, self.tolerance)
            # print "t", frame, begin, end
            start_frame = frame - self.n_frames // 2 + self.offset
            fragment, start_frame, end_frame = self.get_fragment(start_frame, reader)
            del reader
        else:
            # path = path.replace("preproc/con_bcolz/", "")+".K.avi"
            # clip = VideoFileClip(path)
            vid_id = sample["vid_id"]
            path = self.vid_paths[vid_id]
            fps = self.meta[vid_id]["fps"]
            max_frames = self.meta[vid_id]["max_frames"]

            if self.rand_middle_frame: frame = random.randint(begin-self.tolerance, end+self.tolerance)
            else: frame = (end + begin) // 2+random.randint(-self.tolerance, self.tolerance)
            start_frame = frame - self.n_frames // 2 + self.offset
            start_frame = max(start_frame, 0)
            end_frame = min(start_frame + self.n_frames, max_frames-1)
            start_frame = max(end_frame - self.n_frames, 0)

            fragment = extract_video_fragment(path, start_frame, (240, 320), fps, self.n_frames)
            fragment = fragment[...,0]
            # print path
            # sys.exit()
            # fragment = extract_video_fragment()
            # pass

        # print path

        if fragment is None: raise Exception("coulnd't load " + str(path))

        # DEBUG
        # sample["video"] = (np.random.rand(self.n_frames, 288, 704, 3)*256).astype(np.uint8)

        sample["video"] = fragment
        sample["begin"] = start_frame
        sample["end"] = end_frame


    def get_fragment(self, start_frame, reader, push_start=True):
        start_frame = max(start_frame, 0)
        end_frame = min(start_frame + self.n_frames, reader.shape[0])
        if push_start:
            start_frame = max(end_frame - self.n_frames, 0)

        if not self.depth:
            fragment = reader[start_frame: end_frame, ..., 0]
            fragment = fragment.astype("float32")
            if self.rgb:
                fragment += self.rgbbias
        elif not self.rgb:
            fragment = reader[start_frame: end_frame, ..., 1]
            fragment = fragment.astype("float32")
        else:
            fragment = reader[start_frame: end_frame]
            fragment = fragment.astype("float32")
            if self.rgb:
                fragment[..., 0] += self.rgbbias
        return fragment, start_frame, end_frame

