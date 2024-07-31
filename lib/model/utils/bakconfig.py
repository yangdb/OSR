from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# Training options
#

__C.TRAIN = edict()
__C.TRAIN.SET = 'voc'
__C.NEW_CLASSES = 5 #10#10#5#10#5 #############################number of new classes
__C.TRAIN.SQE = None #['5a', '5b']#,'5c','5d']#['10','voc-11']#,'voc-12','voc-13','voc-14']#['5a', '5b','5c','5d'],['2014_coco_a','2014_coco_b']
__C.TRAIN.pdsllam = 5


__C.TRAIN.feadim = 2048
__C.TRAIN.rdc = True 
#['10','voc-1112']#['15','voc-16','voc-17','voc-18','voc-19','voc-20']#['5a','5b']#['15','voc-16']#['10','voc-1112']#['10','voc-11']#,'voc-12']#,'voc-13']#['15','voc-16','voc-17','voc-18','voc-19','voc-20']#None #['10','voc-1112']['15','voc-16']['5a', '5b_5', '5c_5'] #None #['5a'] #None # ['5a'] #None ['5a', '5b_5']

__C.TRAIN.excls = False #True
__C.TRAIN.excp = False #True ## False:Copy RCNN_top True: New RCNN_top
__C.TRAIN.pool = True ## True-extra classifier on the pooled feature; False-on the base feature
__C.TRAIN.bias = True #False


__C.TRAIN.CROP_AUG = True #None
s_num = 1#50 #20 #5
n_num = 'all'
__C.TRAIN.SAVEFLAG =False#True
__C.TRAIN.MASK = False
__C.TRAIN.NUM_SAMPLES = 1 #1 #3 #5
__C.TRAIN.PSE = True#False
__C.TRAIN.pseth = 0.9
__C.TRAIN.isda =True
__C.TRAIN.kg = False #True
if __C.TRAIN.isda:
    __C.TRAIN.excls = True
    __C.TRAIN.excp = True #False ## False:Copy RCNN_top True: New RCNN_top
    __C.TRAIN.pool = True


__C.TRAIN.proto = False #False
__C.TRAIN.CTS = False #True
__C.TRAIN.TMP = 1 #0.07
__C.TRAIN.mixup = False #True

#__C.TRAIN.BG = 'img' #'rand'#'rand'#'rand'#'patch','gray'
ROOT_P ='/mnt/disk7/ydb/Object_train/' #'/data4/'#
__C.TRAIN.ROOT_P = ROOT_P
#__C.TRAIN.CROP_PATH = ROOT_P+'ydb/CPR-IOD/faster-rcnn/data/2007_crops/19_5_new_pad/'#_pad/'

__C.TRAIN.GRAY_AUG = True
__C.TRAIN.GRAY_NUM = 10#200#5#20#25#50#10#10#1#5
__C.TRAIN.GRAY_RATIO = 0.5
__C.TRAIN.BG = 'img' #'rand'#'rand'#'rand'#'patch','gray'
#__C.TRAIN.NEW_PATH = ROOT_P+'ydb/CPR-IOD/faster-rcnn/data/2007_crops/inc-5_5_new_pad/'#ROOT_P+'ydb/CPR-IOD/faster-rcnn/data/2007_crops/inc-tv_5_new_pad/'
__C.TRAIN.GRAY_AUG_NUM_SAMPLES = 1 #3#2#4 ### actually +1 samples
__C.TRAIN.mixobj = True
__C.TRAIN.mixobj_crop = False #True
__C.TRAIN.PSE_NEW = False
__C.TRAIN.pn_ratio = 8#2 #2
__C.TRAIN.mix_lam = 0.5#0.6
__C.TRAIN.mixre = False #True
__C.TRAIN.mixnum = 2


__C.TRAIN.IM_AUG = False #True
__C.TRAIN.CROP_CPGT = True
__C.TRAIN.GRAY_CPGT = True
__C.TRAIN.Filter = False
#__C.TRAIN.SQE = None #['5a', '5b']#,'5c','5d']#['10','voc-11']#,'voc-12','voc-13','voc-14']#['5a', '5b','5c','5d'],['2014_coco_a','2014_coco_b']
#['10','voc-1112']#['15','voc-16','voc-17','voc-18','voc-19','voc-20']#['5a','5b']#['15','voc-16']#['10','voc-1112']#['10','voc-11']#,'voc-12']#,'voc-13']#['15','voc-16','voc-17','voc-18','voc-19','voc-20']#None #['10','voc-1112']['15','voc-16']['5a', '5b_5', '5c_5'] #None #['5a'] #None # ['5a'] #None ['5a', '5b_5']
if not __C.TRAIN.IM_AUG:
    if __C.NEW_CLASSES==1:
        __C.TRAIN.CROP_PATH=ROOT_P+'CPR-IOD/faster-rcnn/data/2007_crops/19_'+str(s_num)+'_new_pad/'#_pad/'
        __C.TRAIN.NEW_PATH = ROOT_P+'CPR-IOD/faster-rcnn/data/2007_crops/inc-tv_'+str(n_num)+'_new_pad/'
        if __C.TRAIN.SQE is not None:
            __C.TRAIN.CROP_PATH=[]
            for sq in __C.TRAIN.SQE[:-1]:
                __C.TRAIN.CROP_PATH.append(ROOT_P+'CPR-IOD/faster-rcnn/data/2007_crops/' + sq+'_' +str(s_num)+ '_new_pad/')
            __C.TRAIN.NEW_PATH = ROOT_P+'CPR-IOD/faster-rcnn/data/2007_crops/'+__C.TRAIN.SQE[-1]+'_'+str(n_num)+'_new_pad/'#voc-16_50_new_pad
    elif __C.NEW_CLASSES==5:
        __C.TRAIN.CROP_PATH=ROOT_P+'CPR-IOD/faster-rcnn/data/2007_crops/15_'+str(s_num)+'_new_pad/'#_pad/'
        #__C.TRAIN.CROP_PATH=ROOT_P+'CPR-IOD/faster-rcnn/crops/15_pad_min/'
        __C.TRAIN.NEW_PATH = ROOT_P+'CPR-IOD/faster-rcnn/data/2007_crops/inc-5_'+str(n_num)+'_new_pad/'
        if __C.TRAIN.SQE is not None:
            __C.TRAIN.CROP_PATH=[]
            for sq in __C.TRAIN.SQE[:-1]:
                __C.TRAIN.CROP_PATH.append(ROOT_P+'CPR-IOD/faster-rcnn/data/2007_crops/' + sq+'_' +str(s_num)+ '_new_pad/')
            __C.TRAIN.NEW_PATH = ROOT_P+'CPR-IOD/faster-rcnn/data/2007_crops/'+__C.TRAIN.SQE[-1]+'_'+str(n_num)+'_new_pad/'
    elif __C.NEW_CLASSES==10:
        __C.TRAIN.CROP_PATH=ROOT_P+'CPR-IOD/faster-rcnn/data/2007_crops/10_'+str(s_num)+'_new_pad/'#_pad/'
        __C.TRAIN.NEW_PATH = ROOT_P+'CPR-IOD/faster-rcnn/data/2007_crops/inc-10_'+str(n_num)+'_new_pad/'
    elif __C.NEW_CLASSES==2:
        if __C.TRAIN.SQE is not None:
            __C.TRAIN.CROP_PATH=[]
            for sq in __C.TRAIN.SQE[:-1]:
                __C.TRAIN.CROP_PATH.append(ROOT_P+'CPR-IOD/faster-rcnn/data/2007_crops/' + sq+'_' +str(s_num)+ '_new_pad/')
            __C.TRAIN.NEW_PATH = ROOT_P+'CPR-IOD/faster-rcnn/data/2007_crops/'+__C.TRAIN.SQE[-1]+'_'+str(n_num)+'_new_pad/'
    if __C.TRAIN.SET=='coco':
        if __C.NEW_CLASSES==40:
            __C.TRAIN.CROP_PATH=ROOT_P+'CPR-IOD/faster-rcnn/data/2007_crops/2014_coco_40_'+str(s_num)+'_new_pad/'#_pad/'
            __C.TRAIN.NEW_PATH = ROOT_P+'CPR-IOD/faster-rcnn/data/2007_crops/2014_coco_inc_40_'+str(n_num)+'_new_pad/'
        elif __C.NEW_CLASSES==20:
            if __C.TRAIN.SQE is not None:
                __C.TRAIN.CROP_PATH=[]
                for sq in __C.TRAIN.SQE[:-1]:
                    __C.TRAIN.CROP_PATH.append(ROOT_P+'CPR-IOD/faster-rcnn/data/2007_crops/' + sq+'_' +str(s_num)+ '_new_pad/')
                __C.TRAIN.NEW_PATH = ROOT_P+'CPR-IOD/faster-rcnn/data/2007_crops/'+__C.TRAIN.SQE[-1]+'_'+str(n_num)+'_new_pad/'
            else:
                __C.TRAIN.CROP_PATH=ROOT_P+'CPR-IOD/faster-rcnn/data/2007_crops/2014_coco_40_'+str(s_num)+'_new_pad/'#_pad/'
                __C.TRAIN.NEW_PATH = ROOT_P+'CPR-IOD/faster-rcnn/data/2007_crops/2014_coco_inc_20_'+str(n_num)+'_new_pad/'
        elif __C.NEW_CLASSES==10:
                __C.TRAIN.CROP_PATH=ROOT_P+'CPR-IOD/faster-rcnn/data/2007_crops/2014_coco_40_'+str(s_num)+'_new_pad/'#_pad/'
                __C.TRAIN.NEW_PATH = ROOT_P+'CPR-IOD/faster-rcnn/data/2007_crops/2014_coco_inc_10_'+str(n_num)+'_new_pad/'
else:
    __C.TRAIN.GRAY_AUG = False
    __C.TRAIN.CROP_AUG = False #None
    __C.TRAIN.CTS = False #True #False #True
    __C.TRAIN.Filter = False


### gray aug for both old and new classes###############
now_class = 20
if __C.TRAIN.SET == 'voc':
    if __C.TRAIN.SQE is not None:
        if __C.NEW_CLASSES!=5:
            old_class = int(__C.TRAIN.SQE[-2].split('-')[-1][-2:]) #[5,10,15],[15,16,17,18,19],[10,11,12,13,14,15,16,17,18,19],[10,12,14,16,18]
        else:
            old_class = 5*(len(__C.TRAIN.SQE)-1)
        now_class = old_class +__C.NEW_CLASSES
elif __C.TRAIN.SET == 'coco':
    old_class = 40
    if __C.TRAIN.SQE is not None:
        old_class = (len(__C.TRAIN.SQE)-1)*20 ### abcd groups
    now_class = old_class +__C.NEW_CLASSES


__C.TRAIN.GRAY_NUM *= now_class #(20-__C.NEW_CLASSES)
__C.TRAIN.LAMBDA = 1 #min(max(0.1,0.1*(now_class-__C.NEW_CLASSES)/__C.NEW_CLASSES), 1.0)


if __C.TRAIN.CROP_AUG == False and __C.TRAIN.GRAY_AUG == True:
    __C.TRAIN.GRAY_PATH = [__C.TRAIN.NEW_PATH, __C.TRAIN.CROP_PATH]
#__C.TRAIN.CROP_AUG=None

__C.TRAIN.BASE_PATH = __C.TRAIN.NEW_PATH#__C.TRAIN.CROP_PATH#__C.TRAIN.NEW_PATH#__C.TRAIN.CROP_PATH#__C.TRAIN.NEW_PATH
__C.TRAIN.GRAY_PATH = __C.TRAIN.CROP_PATH
__C.TRAIN.MIX_PATH = __C.TRAIN.CROP_PATH #[__C.TRAIN.NEW_PATH, __C.TRAIN.CROP_PATH]
__C.TRAIN.MIX_PATH_1 = __C.TRAIN.NEW_PATH
__C.TRAIN.BASE_NUM = 1 #0#1#2  #gray sample gt crop num (for new classes)
__C.TRAIN.TOTAL_GRAY = 5#7 #4   #2 #3 #7#6

#if __C.TRAIN.SET == 'voc':
#    if __C.TRAIN.SQE is None:
#        __C.TRAIN.BASE_NUM = int(__C.TRAIN.TOTAL_GRAY*__C.NEW_CLASSES/20+0.5)
#        if __C.TRAIN.BASE_NUM == 0:
#            __C.TRAIN.BASE_PATH =  __C.TRAIN.CROP_PATH
#            __C.TRAIN.BASE_NUM = 1
#    else:
#        __C.TRAIN.BASE_NUM = int(__C.TRAIN.TOTAL_GRAY*__C.NEW_CLASSES/now_class+0.5)
#        if __C.TRAIN.BASE_NUM == 0:
#            __C.TRAIN.BASE_PATH =  __C.TRAIN.CROP_PATH
#            __C.TRAIN.BASE_NUM = 1
#elif __C.TRAIN.SET == 'coco':
#    if __C.TRAIN.SQE is None:
#        __C.TRAIN.BASE_NUM = int(__C.TRAIN.TOTAL_GRAY*__C.NEW_CLASSES/(40+__C.NEW_CLASSES)+0.5)
#        if __C.TRAIN.BASE_NUM == 0:
#            __C.TRAIN.BASE_PATH =  __C.TRAIN.CROP_PATH
#            __C.TRAIN.BASE_NUM = 1
#    else:
#        __C.TRAIN.BASE_NUM = int(__C.TRAIN.TOTAL_GRAY*__C.NEW_CLASSES/now_class+0.5)
#        if __C.TRAIN.BASE_NUM == 0:
#            __C.TRAIN.BASE_PATH =  __C.TRAIN.CROP_PATH
#            __C.TRAIN.BASE_NUM = 1

__C.TRAIN.GRAY_AUG_NUM_SAMPLES = __C.TRAIN.TOTAL_GRAY-__C.TRAIN.BASE_NUM




__C.threshold_2=False

# Initial learning rate
__C.TRAIN.LEARNING_RATE = 0.001

# Momentum
__C.TRAIN.MOMENTUM = 0.9

# Weight decay, for regularization
__C.TRAIN.WEIGHT_DECAY = 0.0005

# Factor for reducing the learning rate
__C.TRAIN.GAMMA = 0.1

# Step size for reducing the learning rate, currently only support one step
__C.TRAIN.STEPSIZE = [30000]

# Iteration intervals for showing the loss during training, on command line interface
__C.TRAIN.DISPLAY = 10

# Whether to double the learning rate for bias
__C.TRAIN.DOUBLE_BIAS = True

# Whether to initialize the weights with truncated normal distribution
__C.TRAIN.TRUNCATED = False

# Whether to have weight decay on bias as well
__C.TRAIN.BIAS_DECAY = False

# Whether to add ground truth boxes to the pool when sampling regions
__C.TRAIN.USE_GT = False

# Whether to use aspect-ratio grouping of training images, introduced merely for saving
# GPU memory
__C.TRAIN.ASPECT_GROUPING = False

# The number of snapshots kept, older ones are deleted to save space
__C.TRAIN.SNAPSHOT_KEPT = 3

# The time interval for saving tensorflow summaries
__C.TRAIN.SUMMARY_INTERVAL = 180

# Scale to use during training (can list multiple scales)
# The scale is the pixel size of an image's shortest side
__C.TRAIN.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
__C.TRAIN.MAX_SIZE = 1000

# Trim size for input images to create minibatch
__C.TRAIN.TRIM_HEIGHT = 600
__C.TRAIN.TRIM_WIDTH = 600

# Images to use per minibatch
__C.TRAIN.IMS_PER_BATCH = 1

# Minibatch size (number of regions of interest [ROIs])
__C.TRAIN.BATCH_SIZE = 128

# Fraction of minibatch that is labeled foreground (i.e. class > 0)
__C.TRAIN.FG_FRACTION = 0.25

# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH = 0.5

# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.1

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = False#True

# Train bounding-box regressors
__C.TRAIN.BBOX_REG = True

# Overlap required between a ROI and ground-truth box in order for that ROI to
# be used as a bounding-box regression training example
__C.TRAIN.BBOX_THRESH = 0.5

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 5000

# solver.prototxt specifies the snapshot path prefix, this adds an optional
# infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
__C.TRAIN.SNAPSHOT_PREFIX = 'res101_faster_rcnn'
# __C.TRAIN.SNAPSHOT_INFIX = ''

# Use a prefetch thread in roi_data_layer.layer
# So far I haven't found this useful; likely more engineering work is required
# __C.TRAIN.USE_PREFETCH = False

# Normalize the targets (subtract empirical mean, divide by empirical stddev)
__C.TRAIN.BBOX_NORMALIZE_TARGETS = True
# Deprecated (inside weights)
__C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Normalize the targets using "precomputed" (or made up) means and stdevs
# (BBOX_NORMALIZE_TARGETS must also be True)
__C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
__C.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
__C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

# Train using these proposals
__C.TRAIN.PROPOSAL_METHOD = 'gt'

# Make minibatches from images that have similar aspect ratios (i.e. both
# tall and thin or both short and wide) in order to avoid wasting computation
# on zero-padding.

# Use RPN to detect objects
__C.TRAIN.HAS_RPN = True
# IOU >= thresh: positive example
__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
# IOU < thresh: negative example
__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
# If an anchor statisfied by positive and negative conditions set to negative
__C.TRAIN.RPN_CLOBBER_POSITIVES = False
# Max number of foreground examples
__C.TRAIN.RPN_FG_FRACTION = 0.5
# Total number of examples
__C.TRAIN.RPN_BATCHSIZE = 256
# NMS threshold used on RPN proposals
__C.TRAIN.RPN_NMS_THRESH = 0.7
# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TRAIN.RPN_POST_NMS_TOP_N = 2000
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TRAIN.RPN_MIN_SIZE = 8
# Deprecated (outside weights)
__C.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Give the positive RPN examples weight of p * 1 / {num positives}
# and give negatives a weight of (1 - p)
# Set to -1.0 to use uniform example weighting
__C.TRAIN.RPN_POSITIVE_WEIGHT = -1.0
# Whether to use all ground truth bounding boxes for training,
# For COCO, setting USE_ALL_GT to False will exclude boxes that are flagged as ''iscrowd''
__C.TRAIN.USE_ALL_GT = True

# Whether to tune the batch normalization parameters during training
__C.TRAIN.BN_TRAIN = False

#
# Testing options
#
__C.TEST = edict()

# Scale to use during testing (can NOT list multiple scales)
# The scale is the pixel size of an image's shortest side
__C.TEST.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
__C.TEST.MAX_SIZE = 1000

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.NMS = 0.3

# Experimental: treat the (K+1) units in the cls_score layer as linear
# predictors (trained, eg, with one-vs-rest SVMs).
__C.TEST.SVM = False

# Test using bounding-box regressors
__C.TEST.BBOX_REG = True

# Propose boxes
__C.TEST.HAS_RPN = False

# Test using these proposals
__C.TEST.PROPOSAL_METHOD = 'gt'

## NMS threshold used on RPN proposals
__C.TEST.RPN_NMS_THRESH = 0.7
## Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.RPN_PRE_NMS_TOP_N = 6000

## Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RPN_POST_NMS_TOP_N = 300

# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TEST.RPN_MIN_SIZE = 16

# Testing mode, default to be 'nms', 'top' is slower but better
# See report for details
__C.TEST.MODE = 'nms'

# Only useful when TEST.MODE is 'top', specifies the number of top proposals to select
__C.TEST.RPN_TOP_N = 5000

#
# ResNet options
#

__C.RESNET = edict()

# Option to set if max-pooling is appended after crop_and_resize.
# if true, the region will be resized to a square of 2xPOOLING_SIZE,
# then 2x2 max-pooling is applied; otherwise the region will be directly
# resized to a square of POOLING_SIZE
__C.RESNET.MAX_POOL = False

# Number of fixed blocks during training, by default the first of all 4 blocks is fixed
# Range: 0 (none) to 3 (all)
__C.RESNET.FIXED_BLOCKS = 1

#
# MobileNet options
#

__C.MOBILENET = edict()

# Whether to regularize the depth-wise filters during training
__C.MOBILENET.REGU_DEPTH = False

# Number of fixed layers during training, by default the first of all 14 layers is fixed
# Range: 0 (none) to 12 (all)
__C.MOBILENET.FIXED_LAYERS = 5

# Weight decay for the mobilenet weights
__C.MOBILENET.WEIGHT_DECAY = 0.00004

# Depth multiplier
__C.MOBILENET.DEPTH_MULTIPLIER = 1.

#
# MISC
#

# The mapping from image coordinates to feature map coordinates might cause
# some boxes that are distinct in image space to become identical in feature
# coordinates. If DEDUP_BOXES > 0, then DEDUP_BOXES is used as the scale factor
# for identifying duplicate boxes.
# 1/16 is correct for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
__C.DEDUP_BOXES = 1. / 16.

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# For reproducibility
__C.RNG_SEED = 3

# A small number that's used many times
__C.EPS = 1e-14

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))

# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# Name (or path to) the matlab executable
__C.MATLAB = 'matlab'

# Place outputs under an experiments directory
__C.EXP_DIR = 'default'

# Use GPU implementation of non-maximum suppression
__C.USE_GPU_NMS = True

# Default GPU device id
__C.GPU_ID = 0

__C.POOLING_MODE = 'align'#'crop'

# Size of the pooled region after RoI pooling
__C.POOLING_SIZE = 7

# Maximal number of gt rois in an image during Training
__C.MAX_NUM_GT_BOXES = 20

# Anchor scales for RPN
__C.ANCHOR_SCALES = [8,16,32]

# Anchor ratios for RPN
__C.ANCHOR_RATIOS = [0.5,1,2]

# Feature stride for RPN
__C.FEAT_STRIDE = [16, ]

__C.CUDA = False

__C.CROP_RESIZE_WITH_MAX_POOL = True

import pdb
def get_output_dir(imdb, weights_filename):
  """Return the directory where experimental artifacts are placed.
  If the directory does not exist, it is created.

  A canonical path is built using the name from an imdb and a network
  (if not None).
  """
  outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
  if weights_filename is None:
    weights_filename = 'default'
  outdir = osp.join(outdir, weights_filename)
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  return outdir


def get_output_tb_dir(imdb, weights_filename):
  """Return the directory where tensorflow summaries are placed.
  If the directory does not exist, it is created.

  A canonical path is built using the name from an imdb and a network
  (if not None).
  """
  outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'tensorboard', __C.EXP_DIR, imdb.name))
  if weights_filename is None:
    weights_filename = 'default'
  outdir = osp.join(outdir, weights_filename)
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  return outdir


def _merge_a_into_b(a, b):
  """Merge config dictionary a into config dictionary b, clobbering the
  options in b whenever they are also specified in a.
  """
  if type(a) is not edict:
    return

  for k, v in a.items():
    # a must specify keys that are in b
    if k not in b:
      raise KeyError('{} is not a valid config key'.format(k))

    # the types must match, too
    old_type = type(b[k])
    if old_type is not type(v):
      if isinstance(b[k], np.ndarray):
        v = np.array(v, dtype=b[k].dtype)
      else:
        raise ValueError(('Type mismatch ({} vs. {}) '
                          'for config key: {}').format(type(b[k]),
                                                       type(v), k))

    # recursively merge dicts
    if type(v) is edict:
      try:
        _merge_a_into_b(a[k], b[k])
      except:
        print(('Error under config key: {}'.format(k)))
        raise
    else:
      b[k] = v


def cfg_from_file(filename):
  """Load a config file and merge it into the default options."""
  import yaml
  with open(filename, 'r') as f:
    yaml_cfg = edict(yaml.load(f))

  _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
  """Set config keys via list (e.g., from command line)."""
  from ast import literal_eval
  assert len(cfg_list) % 2 == 0
  for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
    key_list = k.split('.')
    d = __C
    for subkey in key_list[:-1]:
      assert subkey in d
      d = d[subkey]
    subkey = key_list[-1]
    assert subkey in d
    try:
      value = literal_eval(v)
    except:
      # handle the case when v is a string literal
      value = v
    assert type(value) == type(d[subkey]), \
      'type {} does not match original type {}'.format(
        type(value), type(d[subkey]))
    d[subkey] = value
