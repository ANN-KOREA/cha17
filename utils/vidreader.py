"""
This module implements all the functions to read a video or a picture
using ffmpeg.
"""

from __future__ import division

import subprocess as sp
import re
import warnings
import logging

logging.captureWarnings(True)

import numpy as np
from moviepy.config import get_setting  # ffmpeg, ffmpeg.exe, etc...
from moviepy.tools import cvsecs

import os


try:
    from subprocess import DEVNULL  # py3k
except ImportError:
    DEVNULL = open(os.devnull, 'wb')


class VideoReader:
    def __init__(self, filename, size, starttime=0.,  print_infos=False, bufsize=None, check_duration=True):

        self.filename = filename
        # infos = ffmpeg_parse_infos(filename, print_infos, check_duration)
        # # self.fps = infos['video_fps']
        # self.size = infos['video_size']
        # # self.duration = infos['video_duration']
        # # self.ffmpeg_duration = infos['duration']
        # self.nframes = infos['video_nframes']
        # self.infos = infos
        self.size = size[1], size[0]

        self.depth = 3

        w, h = self.size
        if bufsize is None:
            bufsize = self.depth * w * h + 100

        self.bufsize = bufsize
        self.initialize(starttime)
        self.pos = 0

        self.nbytes = self.depth * w * h
        self.shape = (h, w, self.nbytes // (w * h))

    def initialize(self, starttime=0.):
        """Opens the file, creates the pipe. """

        if starttime != 0:
            offset = min(1, starttime)
            i_arg = ['-ss', "%.06f" % (starttime - offset),
                     '-i', self.filename,
                     '-ss', "%.06f" % offset]
        else:
            i_arg = ['-i', self.filename]
            
        # print get_setting("FFMPEG_BINARY")
        cmd = ([get_setting("FFMPEG_BINARY")] + i_arg +
               ['-loglevel', 'error',
                '-f', 'image2pipe',
                "-pix_fmt", "rgb24",
                '-vcodec', 'rawvideo', '-'])

        popen_params = {"bufsize": self.bufsize,
                        "stdout": sp.PIPE,
                        "stderr": sp.PIPE,
                        "stdin": DEVNULL}

        if os.name == "nt":
            popen_params["creationflags"] = 0x08000000

        self.proc = sp.Popen(cmd, **popen_params)

    def read_frame(self):
        s = self.proc.stdout.read(self.nbytes)
        if len(s) != self.nbytes:
            raise IOError(("VideoReader error: failed to read the first frame of "
                           "video file %s. That might mean that the file is "
                           "corrupted. That may also mean that you are using "
                           "a deprecated version of FFMPEG. On Ubuntu/Debian "
                           "for instance the version in the repos is deprecated. "
                           "Please update to a recent version from the website.") % (
                              self.filename))
        else:
            result = np.fromstring(s, dtype='uint8')
            result.shape = self.shape  # reshape((h, w, len(s)//(w*h)))

        return result

    def close(self):
        if hasattr(self, 'proc'):
            self.proc.terminate()
            self.proc.stdout.close()
            self.proc.stderr.close()
            del self.proc

    def __del__(self):
        self.close()


def extract_video_fragment(path, start_frame, shape, fps, n_frames, skip_frames=0):
    fps = float(fps)
    t_start = start_frame / fps

    reader = VideoReader(path, shape, starttime=t_start)
    vid = np.empty((n_frames,) + shape + (3,), np.uint8)

    dst_idx = 0
    src_idx = 0
    while dst_idx != n_frames:
        img = reader.read_frame()
        # try: img = reader.read_frame()
        # except: return None
        if img is None: return None
        if src_idx % (skip_frames + 1) == 0:
            vid[dst_idx] = img
            dst_idx += 1
        src_idx += 1

    reader.close()
    return vid


def ffmpeg_parse_infos(filename, print_infos=False, check_duration=True):
    """Get file infos using ffmpeg.

    Returns a dictionnary with the fields:
    "video_found", "video_fps", "duration", "video_nframes",
    "video_duration", "audio_found", "audio_fps"

    "video_duration" is slightly smaller than "duration" to avoid
    fetching the uncomplete frames at the end, which raises an error.

    """

    # open the file in a pipe, provoke an error, read output
    cmd = [get_setting("FFMPEG_BINARY"), "-i", filename]

    popen_params = {"bufsize": 10 ** 5,
                    "stdout": sp.PIPE,
                    "stderr": sp.PIPE,
                    "stdin": DEVNULL}

    if os.name == "nt":
        popen_params["creationflags"] = 0x08000000

    proc = sp.Popen(cmd, **popen_params)

    proc.stdout.readline()
    proc.terminate()
    infos = proc.stderr.read().decode('utf8')
    del proc

    if print_infos:
        # print the whole info text returned by FFMPEG
        print(infos)

    lines = infos.splitlines()
    if "No such file or directory" in lines[-1]:
        raise IOError(("MoviePy error: the file %s could not be found !\n"
                       "Please check that you entered the correct "
                       "path.") % filename)

    result = dict()

    # get duration (in seconds)
    result['duration'] = None

    if check_duration:
        try:
            keyword = 'Duration: '
            line = [l for l in lines if keyword in l][0]
            match = re.findall("([0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9])", line)[0]
            result['duration'] = cvsecs(match)
        except:
            raise IOError(("MoviePy error: failed to read the duration of file %s.\n"
                           "Here are the file infos returned by ffmpeg:\n\n%s") % (
                              filename, infos))

    # get the output line that speaks about video
    lines_video = [l for l in lines if ' Video: ' in l]

    result['video_found'] = (lines_video != [])

    if result['video_found']:

        try:
            line = lines_video[0]

            # get the size, of the form 460x320 (w x h)
            match = re.search(" [0-9]*x[0-9]*(,| )", line)
            s = list(map(int, line[match.start():match.end() - 1].split('x')))
            result['video_size'] = s
        except:
            raise (("MoviePy error: failed to read video dimensions in file %s.\n"
                    "Here are the file infos returned by ffmpeg:\n\n%s") % (
                       filename, infos))

        # get the frame rate. Sometimes it's 'tbr', sometimes 'fps', sometimes
        # tbc, and sometimes tbc/2...
        # Current policy: Trust tbr first, then fps. If result is near from x*1000/1001
        # where x is 23,24,25,50, replace by x*1000/1001 (very common case for the fps).

        try:
            match = re.search("( [0-9]*.| )[0-9]* tbr", line)
            tbr = float(line[match.start():match.end()].split(' ')[1])
            result['video_fps'] = tbr

        except:
            match = re.search("( [0-9]*.| )[0-9]* fps", line)
            result['video_fps'] = float(line[match.start():match.end()].split(' ')[1])

        # It is known that a fps of 24 is often written as 24000/1001
        # but then ffmpeg nicely rounds it to 23.98, which we hate.
        coef = 1000.0 / 1001.0
        fps = result['video_fps']
        for x in [23, 24, 25, 30, 50]:
            if (fps != x) and abs(fps - x * coef) < .01:
                result['video_fps'] = x * coef

        if check_duration:
            result['video_nframes'] = int(result['duration'] * result['video_fps']) + 1
            result['video_duration'] = result['duration']
        else:
            result['video_nframes'] = 1
            result['video_duration'] = None
            # We could have also recomputed the duration from the number
            # of frames, as follows:
            # >>> result['video_duration'] = result['video_nframes'] / result['video_fps']

    lines_audio = [l for l in lines if ' Audio: ' in l]

    result['audio_found'] = lines_audio != []

    if result['audio_found']:
        line = lines_audio[0]
        try:
            match = re.search(" [0-9]* Hz", line)
            result['audio_fps'] = int(line[match.start() + 1:match.end()])
        except:
            result['audio_fps'] = 'unknown'

    return result
