import multiprocessing as mp


def release_lock(lock):
    try: lock.release()
    except: pass