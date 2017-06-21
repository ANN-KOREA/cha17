from abc import abstractmethod
import numpy as np
import os
import multiprocessing as mp
from signal import signal, SIGINT, SIGABRT, SIGKILL, SIGTERM
from uuid import uuid4
from time import sleep, time
import traceback
import random

from utils.mptools import release_lock
from utils import remove_file_if_exists, TRAIN


class BaseLoader(object):
    ids = None
    predicting = False

    @abstractmethod
    def prepare(self): pass
    @abstractmethod
    def load_sample(self, id_): pass
    @abstractmethod
    def sample_ids(self, set, size, offset): pass

    def chunk_generator(self, n_chunks, chunk_size, set): pass

    def __init__(self, inputs=None, outputs=None, preprocessors=None):
        self.inputs = {} if inputs is None else inputs
        self.outputs = {} if outputs is None else outputs
        self.preprocessors = [] if preprocessors is None else preprocessors
        self.io_data = dict(self.inputs, **self.outputs) # combine dicts
        self.loader_job = None
        self.uuid = str(uuid4())

    def start(self, n_jobs=1, **kwargs):
        self.use_multiprocessing = n_jobs > 0
        self.n_jobs = n_jobs

        if self.use_multiprocessing:
            os.setpgrp() # create new process group, become its leader
            # import atexit
            # atexit.register(self.clean)
            signal(SIGINT, self.terminate)

            self.read_lock = mp.Lock()
            self.write_lock = mp.Lock()

            self.chunk_generator = self._chunk_generator_mp
        else:
            self.chunk_generator = self._chunk_generator_no_mp

        print "Preparing %s.." % self.__class__.__name__
        t0 = time()
        self.prepare(**kwargs)
        print "  time: %fs" % (time() - t0)
        for i, prep in enumerate(self.preprocessors):
            if hasattr(prep, "prepare"):
                print "Preparing %s.."%prep.__class__.__name__
                t0 = time()
                prep.prepare(self)
                print "  time: %fs" % (time() - t0)
        self.preprocessors = [p for p in self.preprocessors if hasattr(p, "process")]

    def _chunk_generator_mp(self, n_chunks, chunk_size, set):
        if n_chunks == "all":
            n_chunks = int(np.ceil(len(self.ids[set]) / float(chunk_size)))
            print "Going through ALL", set, "samples"

        self.chunk_size = chunk_size
        self.set = set
        self.n_chunks = n_chunks

        random.seed()
        np.random.seed()
        random.shuffle(self.ids[TRAIN])

        self.read_lock.acquire()  # data is not ready yet !
        self.write_lock.acquire()

        #initialize chunk
        self.chunk = [{},{}] #2 chunks: 1 working chunk, 1 ready chunk
        for i, chunk in enumerate(self.chunk):
            for tag, io in self.io_data.items():
                chunk[tag] = np.memmap("/dev/shm/baseloader_%s_%i_%s" % (tag, i, self.uuid),
                                     mode="w+", dtype=io.dtype, shape=(chunk_size,)+io.shape)

        if self.loader_job is not None: self.loader_job.terminate()
        self.loader_job = mp.Process(target=self.__loader_process, args=())
        self.loader_job.start()

        release_lock(self.write_lock) # start writing
        for chunk_idx in range(n_chunks):
            self.read_lock.acquire()  # wait for working chunk switch to ready
            release_lock(self.write_lock) # start writing on new working chunk
            yield self.chunk[chunk_idx%2] # read ready chunk

        self.loader_job.join(timeout=5)
        self.terminate()

    def __loader_process(self):
        try: self._loader_process()
        except Exception as e: print e
        finally: self.terminate()

    def _loader_process(self):
        # seed = random.randint(-999999999, 999999999)
        # random.seed()
        # np.random.seed()
        self.pool = mp.Pool(self.n_jobs, initializer=_initialize_pool,
                            initargs=(self.load_sample, self.preprocessors, self.chunk))

        self.write_lock.acquire() # wait until writing is possible
        for chunk_idx in range(self.n_chunks):
            self._load_chunk_mp(chunk_idx)
            release_lock(self.read_lock) # reading is now possible
            self.write_lock.acquire() # wait until writing is possible
        # print "loader_process finished"
        self.pool.close()
        self.pool.terminate()
        self.pool.join()

    def _load_chunk_mp(self, chunk_idx):
        chunk_ids = self.sample_ids(self.set, self.chunk_size, chunk_idx * self.chunk_size)

        work_chunk = self.chunk[chunk_idx%2]
        res = [self.pool.apply_async(_process_sample, (i, chunk_ids[i], chunk_idx))
                   for i in range(self.chunk_size)]
        # for r in res: r.wait()
        for r in res:
            if not r.get(): self.terminate()

        # flush memmaps
        for tag in self.io_data.keys(): work_chunk[tag].flush()

    def _chunk_generator_no_mp(self, n_chunks, chunk_size, set):
        self.chunk_size = chunk_size
        self.set = set

        random.seed()
        np.random.seed()
        random.shuffle(self.ids[TRAIN])

        #initialize chunk
        self.chunk = {}
        for tag, io in self.io_data.items():
            self.chunk[tag] = np.empty((chunk_size,)+io.shape, io.dtype)

        for chunk_idx in range(n_chunks):
            yield self._load_chunk_no_mp(chunk_idx)

    def _load_chunk_no_mp(self, chunk_idx):
        chunk_ids = self.sample_ids(self.set, self.chunk_size, chunk_idx*self.chunk_size)

        for i, id_ in enumerate(chunk_ids):
            smpl = self.load_sample(id_)
            for prep in self.preprocessors:
                # t0 = time()
                prep.process(smpl)
                # print prep.__class__.__name__, time() - t0
            for tag in self.io_data.keys(): self.chunk[tag][i] = smpl[tag]

        return self.chunk

    def clean(self):
        # remove shared memory
        if hasattr(self, "chunk") and self.chunk is not None and self.use_multiprocessing:
            for tag in self.io_data.keys():
                for i in range(2): remove_file_if_exists(self.chunk[i][tag].filename)

    def terminate(self, sig=None, frame=None):
        try:
            release_lock(self.read_lock)
            release_lock(self.write_lock)
            if hasattr(self, "pool"):
                try: self.pool.terminate()
                except: pass
            try: self.loader_job.join(timeout=1)
            except: pass
            try: self.loader_job.terminate()
            except: pass
        finally:
            self.clean()
            if sig is not None:
                print "BaseLoader process terminated with signal", sig
                # traceback.print_stack(frame)
                os.killpg(0, sig)
                sleep(1) # give some time until we escalate to kill -9
                os.killpg(0, SIGKILL)


def _initialize_pool(load_sample, preprocessors, chunk):
    global __load_sample, __preprocessors, __chunk
    __load_sample = load_sample
    __preprocessors = preprocessors
    __chunk = chunk
    # print time()
    # import scipy
    # scipy.random.seed()


def _process_sample(sample_idx, id_, chunk_idx):
    global __load_sample, __preprocessors, __chunk

    seed = random.randint(0, 1000000000)
    np.random.seed(seed)
    # print seed, np.random.rand()

    try:
        smpl = __load_sample(id_)
        # smpl["smpl_idx"] = sample_idx
        for prep in __preprocessors:
            # t0 = time()
            prep.process(smpl)
            # print prep.__class__.__name__, time() - t0
            # if "Video" in prep.__class__.__name__:
            #     print prep.__class__.__name__, time() - t0, smpl["end"]-smpl["begin"], smpl["path"]

        work_chunk = __chunk[chunk_idx % 2]

        for tag in work_chunk.keys():
            work_chunk[tag][sample_idx] = smpl[tag]
        return True
    except:
        print "Error in sample preprocessor"
        traceback.print_exc()
        return False
