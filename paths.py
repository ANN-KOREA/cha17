import socket
import os

host = socket.gethostname()


#### CHANGE THE SOURCE VIDEO PATHS HERE:
CON_VID_TRAIN = "/data/lpigou/congd/train/"
CON_VID_VALID = "/data/lpigou/congd/valid/"
CON_VID_TEST = "/data/lpigou/congd/test/"

#### CHANGE THE PREPROCESSING DATA PATH HERE:
CON_PREP2 = "/data/lpigou/congd/preproc/depth_prep/"







#### DONT CHANGE:



CON_LBL_TRAIN = "./data/train.txt"
CON_LBL_VALID = "./data/valid.txt"

if not os.path.exists(CON_PREP2): os.mkdir(CON_PREP2)

CON_VID_TRAIN += "*/*.avi"
CON_VID_VALID += "*/*.avi"
CON_VID_TEST += "*/*.avi"