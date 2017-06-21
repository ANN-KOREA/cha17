import socket
import os

host = socket.gethostname()

CON_LBL_TRAIN = "./data/train.txt"
CON_LBL_VALID = "./data/valid.txt"

CON_VID_TRAIN = "/mnt/ssd/congd/train/"
CON_VID_VALID = "/mnt/ssd/congd/valid/"
CON_VID_TEST = "/mnt/ssd/congd/test/"

if host == "l":
    CON_PREP = "/mnt/ssd/congd/preproc/con_bcolz/"
    CON_PREP2 = "/mnt/ssd/congd/preproc/depth_prep/"
    if not os.path.exists(CON_PREP): os.mkdir(CON_PREP)
else:
    CON_VID_TRAIN = "/data/lpigou/congd/train/"
    CON_VID_VALID = "/data/lpigou/congd/valid/"
    CON_VID_TEST = "/data/lpigou/congd/test/"
    CON_PREP = "/data/lpigou/congd/preproc/con_bcolz/"
    CON_PREP2 = "/data/lpigou/congd/preproc/depth_prep/"

CON_VID_TRAIN += "*/*.avi"
CON_VID_VALID += "*/*.avi"
CON_VID_TEST += "*/*.avi"