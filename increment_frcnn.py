# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import argparse
import pprint
import pdb
import time
import random
import copy

import torch
from torch.autograd import Variable
import torch.nn as nn

from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader as roibatchLoader_org
from roi_data_layer.roibatchLoader import roibatchLoaderinc as roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.utils.net_utils import adjust_learning_rate, save_checkpoint, clip_gradient

#from model.faster_rcnn.vgg16 import vgg16
#from model.faster_rcnn.vgg16_inc_bbox_distil import vgg16_inc
from model.faster_rcnn.resnet_cls import resnet
#from model.faster_rcnn.resnet_inc_bbox_distil import resnet_inc_bbox_distil

from model.faster_rcnn.resnet_inc_bbox_distil_residual_cts import resnet_inc_bbox_distil
from model.faster_rcnn.resnet_split_clsbox_test_residual_cts import resnet_split_clsbox_test
#from model.faster_rcnn.resnet_residual import resnet as resnet_residual
from model.utils.config import cfg

import cv2
from model.utils.blob import im_list_to_blob

import numpy as np
import cv2
from model.utils.augs import seq, superpixel

torch.manual_seed(cfg.RNG_SEED)
np.random.seed(cfg.RNG_SEED)
random.seed(cfg.RNG_SEED)  ##
os.environ['PYTHONHASHSEED'] = str(cfg.RNG_SEED)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.cuda.manual_seed(cfg.RNG_SEED)
torch.cuda.manual_seed_all(cfg.RNG_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.enabled = False


def worker_init_fn(worker_id):
    np.random.seed(int(cfg.RNG_SEED))
    random.seed(cfg.RNG_SEED+worker_id)

def merge_image(image_list, x_num=2, y_num=2, csize=100): ### y:horizon num x:vertical
    image_num = len(image_list)
    # frame = np.zeros((720, 1080), dtype=np.uint8)
    new_image_list = []
    x_image_list = []
    if image_num % x_num == 0:
        # frame = np.zeros((720, 1080), dtype=np.uint8)
        # image = cv2.imread("10005001_1618537982.jpg")
        for image in image_list:
            frame = cv2.resize(image, (csize, csize), interpolation=cv2.INTER_AREA)  # 每张图片的大小可自行resize
            new_image_list.append(frame)
        for x in range(x_num):
            htich = np.hstack([i for i in new_image_list[y_num * x:y_num * (x + 1)]])
            x_image_list.append(htich)
        vtich = np.vstack(x_image_list)
    else:  # 在这里如果图片列表是单数的话，你可以填充一个空白图片
        pass
    return vtich


def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)
    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors)
def getFile(path, fileList, cls_list=None):
    fList = os.listdir(path) #将指定目录内的文件以列表格式输出
    # os.chdir(path)
    docuPath = path #获取根路径
    # fileList = []
    for f in fList: #对目录内的文件进行遍历
        if os.path.isdir(os.path.join(docuPath,f)): #判断文件类型是否为目录
            if cls_list is not None:
                cls_list.append(f)
            getFile(os.path.join(docuPath,f), fileList) #若是目录，递归运行此函数，继续进行遍历。
        else:
            fl = os.path.join(docuPath,f) #若不是目录，则结合文件名和根路径获得文件的绝对路径
            fileList.append(fl)




def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='vgg16', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=20, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)
  parser.add_argument('--load_model', dest='load_model',
                      help='directory to load model', default="",
                      type=str)
  parser.add_argument('--expert_model', dest='expert_model',
                      help='directory to load model', default=None,
                      type=str)
  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="models_inc",
                      type=str)
  parser.add_argument('--prototype', dest='prototype',
                      help='directory to load prototype', default="",
                      type=str)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=0, type=int)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--first', dest='first',
                      help='add the second, third, ...class',
                      action='store_true')
  parser.add_argument('--base_model', dest='base_model',
                      help='directory to load base sqe model', default=None,
                      type=str)
  parser.add_argument('--trained_residual_model', dest='trained_residual_model',default='',type=str)
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')

# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.001, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
# log and diaplay
  parser.add_argument('--use_tfb', dest='use_tfboard',
                      help='whether use tensorboard',
                      action='store_true')

  args = parser.parse_args()
  return args


class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    #self.rand_num_view= torch.Tensor(list(range(self.num_per_batch))).int()#####[0,1,2...,1748]
    #print(self.rand_num_view)
    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

if __name__ == '__main__':

  args = parse_args()
  if args.first:
      from model.faster_rcnn.resnet_cls import resnet
  else:
      from model.faster_rcnn.resnet import resnet
  print('Called with args:')
  print(args)

  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_0712_incre":
      args.imdb_name = "voc_2007_trainval_incre+voc_2012_trainval_incre"
      args.imdbval_name = "voc_2007_test_incre"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_trainval+voc_2012_trainval'
      args.imdbval_name_org = "voc_2007_test"
  elif args.dataset == "pascal_voc_07_incre":
      args.imdb_name = "voc_2007_trainval_incre"
      args.imdbval_name = "voc_2007_test_incre"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_trainval'
      args.imdbval_name_org = "voc_2007_test"
      args.imdb_name_expert = "voc_2007_1_train_tv"  ########## last new class

  elif args.dataset == "pascal_voc_07_incre_re":
      args.imdb_name = "voc_2007_1_incre_re"
      args.imdbval_name = "voc_2007_test_incre"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_trainval'
      args.imdbval_name_org = "voc_2007_test"
      args.imdb_name_expert = "voc_2007_1_train_tv"  ########## last new class
  elif args.dataset == "pascal_voc_07_15_re":
      args.imdb_name = "voc_2007_5_incre_re"
      args.imdbval_name = "voc_2007_15_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_15_train'
      args.imdbval_name_org = "voc_2007_15_test"
      args.imdb_name_expert = "voc_2007_5_train_expert"
  elif args.dataset == "pascal_voc_07_10_re":
      args.imdb_name = "voc_2007_10_incre_re"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_10_train'
      args.imdbval_name_org = "voc_2007_test"
      args.imdb_name_expert = "voc_2007_10_train_expert"

  elif args.dataset == "pascal_voc_07_15":
      args.imdb_name = "voc_2007_5_incre"
      args.imdbval_name = "voc_2007_15_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_15_train'
      args.imdbval_name_org = "voc_2007_15_test"
      args.imdb_name_expert = "voc_2007_5_train_expert"
  elif args.dataset == "pascal_voc_07_15_no":
      args.imdb_name = "voc_2007_5_incre_nonoverlap"
      args.imdbval_name = "voc_2007_15_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_15_train_nonoverlap'
      args.imdbval_name_org = "voc_2007_15_test"
      args.imdb_name_expert = "voc_2007_5_train_expert"
  elif args.dataset == "pascal_voc_07_19_no":
      args.imdb_name = "voc_2007_1_incre_nonoverlap"
      args.imdbval_name = "voc_2007_19_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_19_train_nonoverlap'
      args.imdbval_name_org = "voc_2007_19_test"
      args.imdb_name_expert = "voc_2007_1_train_expert"
  elif args.dataset == "pascal_voc_07_10_no":
      args.imdb_name = "voc_2007_10_incre_nonoverlap"
      args.imdbval_name = "voc_2007_10_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_10_train_nonoverlap'
      args.imdbval_name_org = "voc_2007_10_test"
      args.imdb_name_expert = "voc_2007_10_train_expert"
  elif args.dataset == "pascal_voc_07_15_15_plant":
      args.imdb_name = "voc_2007_15_plant"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_15_train'
      args.imdbval_name_org = "voc_2007_test"
      args.imdb_name_expert = "voc_2007_1_train_plant"  ########## last new class
  elif args.dataset == "pascal_voc_07_11_12":
      args.imdb_name = "voc_2007_11_12_table_dog"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_10_train'
      # args.imdbval_name_org = "voc_2007_test"
      # args.imdb_name_expert = "voc_2007_2_table_dog"  ########## last new class
      # args.imdb_name_base = 'voc_2007_10_train'
      # args.imdb_name_last_expert ="voc_2007_10_train"
  elif args.dataset == "pascal_voc_07_13_14":
      args.imdb_name = "voc_2007_13_14_horse_bike"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_11_12_table_dog'
  elif args.dataset == "pascal_voc_07_15_16":
      args.imdb_name = "voc_2007_15_16_person_plant"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_13_14_horse_bike'
  elif args.dataset == "pascal_voc_07_17_18":
      args.imdb_name = "voc_2007_17_18_sheep_sofa"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_15_16_person_plant'
  elif args.dataset == "pascal_voc_07_19_20":
      args.imdb_name = "voc_2007_19_20_train_tv"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_17_18_sheep_sofa'
  elif args.dataset == "pascal_voc_07_15_16_sheep":
      args.imdb_name = "voc_2007_16_sheep"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_15_plant'
      args.imdbval_name_org = "voc_2007_test"
      args.imdb_name_expert = "voc_2007_1_train_sheep"  ########## last new class
      args.imdb_name_base = 'voc_2007_15_train'
      args.imdb_name_last_expert ="voc_2007_1_train_plant"
  elif args.dataset == "pascal_voc_07_15_17_sofa":
      args.imdb_name = "voc_2007_17_sofa"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_16_sheep'
      args.imdbval_name_org = "voc_2007_test"
      args.imdb_name_expert = "voc_2007_1_train_sofa"  ########## last new class
      args.imdb_name_base = 'voc_2007_15_train'
      args.imdb_name_last_expert ="voc_2007_1_train_sheep"
  elif args.dataset == "pascal_voc_07_15_18_train":
      args.imdb_name = "voc_2007_18_train"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_17_sofa'
      args.imdbval_name_org = "voc_2007_test"
      args.imdb_name_expert = "voc_2007_1_train_train"  ########## last new class
      args.imdb_name_base = 'voc_2007_15_train'
      args.imdb_name_last_expert ="voc_2007_1_train_sofa"
  elif args.dataset == "pascal_voc_07_15_19_tv":
      args.imdb_name = "voc_2007_19_tv"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_18_train'
      args.imdbval_name_org = "voc_2007_test"
      args.imdb_name_expert = "voc_2007_1_train_tv"  ########## last new class
      args.imdb_name_base = 'voc_2007_15_train'
      args.imdb_name_last_expert ="voc_2007_1_train_train"
  elif args.dataset == "pascal_voc_inc_sqe":#['diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
      args.imdb_name = "voc_2007_15_pottedplant"#'voc_2007_10_table'#"voc_2007_12_horse"#"voc_2007_15_pottedplant"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org ='voc_2007_10_train'#'voc_2007_11_dog'#"voc_2007_10_table"#"voc_2007_13_motorbike"# 'voc_2007_12_horse'#'voc_2007_10_train'
      args.imdb_name_expert = "voc_2007_1_train"########## last new class
  elif args.dataset == "pascal_voc_07_10":
      args.imdb_name = "voc_2007_10_incre"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_10_train'
      args.imdbval_name_org = "voc_2007_test"
      args.imdb_name_expert = "voc_2007_10_train_expert"
  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      args.imdbval_name = "coco_2014_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "coco_40_train":
      args.imdb_name = "coco_40_train_inc"
      args.imdbval_name = "coco_2014_minival"
      args.imdb_name_org = 'coco_40_train_base'
      args.imdb_name_expert = "coco_40_train_expert"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "coco_2_train":
      args.imdb_name = "coco_2_train_inc"
      args.imdbval_name = "coco_2014_minival"
      args.imdb_name_org = 'coco_40_train_base'
      args.imdb_name_expert = "coco_40_train_expert"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "coco_5_train":
      args.imdb_name = "coco_5_train_inc"
      args.imdbval_name = "coco_2014_minival"
      args.imdb_name_org = 'coco_40_train_base'
      args.imdb_name_expert = "coco_40_train_expert"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "coco_10_train":
      args.imdb_name = "coco_10_train_inc"
      args.imdbval_name = "coco_2014_minival"
      args.imdb_name_org = 'coco_40_train_base'
      args.imdb_name_expert = "coco_40_train_expert"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "coco_20_train":
      args.imdb_name = "coco_20_train_inc"
      args.imdbval_name = "coco_2014_minival"
      args.imdb_name_org = 'coco_60_train_base'
      args.imdb_name_expert = "coco_40_train_expert"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "coco_1_train":
      args.imdb_name = "coco_1_train_inc"
      args.imdbval_name = "coco_2014_minival"
      args.imdb_name_org = 'coco_40_train_base'
      args.imdb_name_expert = "coco_40_train_expert"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "coco_5_train":
      args.imdb_name = "coco_5_train_inc"
      args.imdbval_name = "coco_2014_minival"
      args.imdb_name_org = 'coco_40_train_base'
      args.imdb_name_expert = "coco_40_train_expert"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "coco_10_train":
      args.imdb_name = "coco_10_train_inc"
      args.imdbval_name = "coco_2014_minival"
      args.imdb_name_org = 'coco_40_train_base'
      args.imdb_name_expert = "coco_40_train_expert"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "coco_4020_train":
      args.imdb_name = "coco_4020_train_inc"
      args.imdbval_name = "coco_2014_minival"
      args.imdb_name_org = 'coco_40_train_base'
      args.imdb_name_expert = "coco_40_train_expert"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
  elif args.dataset == "vg":
      # train sizes: train, smalltrain, minitrain
      # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "pascal_voc_07_1":
      args.imdb_name = "voc_2007_1_train"
      args.imdbval_name = "voc_2007_test_sqe_1"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_07_10_10_table":
      args.imdb_name = "voc_2007_10_10_table"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_10_train'
      args.imdbval_name_org = "voc_2007_test"
      args.imdb_name_expert = "voc_2007_1_train_table"  ########## last new class
  elif args.dataset == "pascal_voc_07_10_11_dog":
      args.imdb_name = "voc_2007_10_11_dog"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_10_10_table'
      args.imdbval_name_org = "voc_2007_test"
      args.imdb_name_expert = "voc_2007_1_train_dog"  ########## last new class
  elif args.dataset == "pascal_voc_07_10_12_horse":
      args.imdb_name = "voc_2007_10_12_horse"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_10_11_dog'
      args.imdbval_name_org = "voc_2007_test"
      args.imdb_name_expert = "voc_2007_1_train_horse"  ########## last new class
  elif args.dataset == "pascal_voc_07_10_13_motorbike":
      args.imdb_name = "voc_2007_10_13_motorbike"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_10_12_horse'
      args.imdbval_name_org = "voc_2007_test"
      args.imdb_name_expert = "voc_2007_1_train_motorbike"  ########## last new class
  elif args.dataset == "pascal_voc_07_10_14_person":
      args.imdb_name = "voc_2007_10_14_person"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_10_13_motorbike'
      args.imdbval_name_org = "voc_2007_test"
      args.imdb_name_expert = "voc_2007_1_train_person"  ########## last new class
  elif args.dataset == "pascal_voc_07_10_15_plant":
      args.imdb_name = "voc_2007_10_15_pottedplant"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_10_14_person'
      args.imdbval_name_org = "voc_2007_test"
      args.imdb_name_expert = "voc_2007_1_train_plant"  ########## last new class
  elif args.dataset == "pascal_voc_07_10_16_sheep":
      args.imdb_name = "voc_2007_10_16_sheep"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_10_15_pottedplant'
      args.imdbval_name_org = "voc_2007_test"
      args.imdb_name_expert = "voc_2007_1_train_sheep"  ########## last new class
  elif args.dataset == "pascal_voc_07_10_17_sofa":
      args.imdb_name = "voc_2007_10_17_sofa"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_10_16_sheep'
      args.imdbval_name_org = "voc_2007_test"
      args.imdb_name_expert = "voc_2007_1_train_sofa"  ########## last new class
  elif args.dataset == "pascal_voc_07_10_18_train":
      args.imdb_name = "voc_2007_10_18_train"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_10_17_sofa'
      args.imdbval_name_org = "voc_2007_test"
      args.imdb_name_expert = "voc_2007_1_train_train"  ########## last new class
  elif args.dataset == "pascal_voc_07_10_19_tv":
      args.imdb_name = "voc_2007_10_19_tvmonitor"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_10_18_train'
      args.imdbval_name_org = "voc_2007_test"
      args.imdb_name_expert = "voc_2007_1_train_tv"  ########## last new class
  elif args.dataset == "pascal_voc_07_5_b":
      args.imdb_name = "voc_2007_5_b"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_5_a'
      args.imdbval_name_org = "voc_2007_test"
      args.imdb_name_expert = "voc_2007_1_5b"
  elif args.dataset == "pascal_voc_07_5_c":
      args.imdb_name = "voc_2007_5_c"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_5_b'
      args.imdbval_name_org = "voc_2007_test"
      args.imdb_name_expert = "voc_2007_1_5c"
  elif args.dataset == "pascal_voc_07_5_d":
      args.imdb_name = "voc_2007_5_d"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_5_c'
      args.imdbval_name_org = "voc_2007_test"
      args.imdb_name_expert = "voc_2007_1_5d"
  elif args.dataset == "pascal_voc_07_19_inc_plant":
      args.imdb_name = "voc_2007_19_plant_inc"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_19_ex_plant'
      args.imdbval_name_org = "voc_2007_test"
      args.imdb_name_expert = "voc_2007_19_1_plant_inc"
  elif args.dataset == "pascal_voc_07_19_inc_sheep":
      args.imdb_name = "voc_2007_19_sheep_inc"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_19_ex_sheep'
      args.imdbval_name_org = "voc_2007_test"
      args.imdb_name_expert = "voc_2007_19_1_sheep_inc"
  elif args.dataset == "pascal_voc_07_19_inc_sofa":
      args.imdb_name = "voc_2007_19_sofa_inc"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_19_ex_sofa'
      args.imdbval_name_org = "voc_2007_test"
      args.imdb_name_expert = "voc_2007_19_1_sofa_inc"
  elif args.dataset == "pascal_voc_07_19_inc_train":
      args.imdb_name = "voc_2007_19_train_inc"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_19_ex_train'
      args.imdbval_name_org = "voc_2007_test"
      args.imdb_name_expert = "voc_2007_19_1_train_inc"
  elif args.dataset == "coco_14_train_b":
      args.imdb_name = "coco_14_train_b"
      args.imdbval_name = "coco_2014_minival"
      args.imdb_name_org = 'coco_14_train_a'
      args.imdb_name_expert = "coco_14_train_b_expert"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "coco_b":
      args.imdb_name = "coco_b"
      args.imdbval_name = "coco_2014_minival"
      args.imdb_name_org = 'coco_a'
      args.imdb_name_expert = "coco_b_expert"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "coco_c":
      args.imdb_name = "coco_c"
      args.imdbval_name = "coco_2014_minival"
      args.imdb_name_org = 'coco_b'
      args.imdb_name_expert = "coco_c_expert"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "coco_d":
      args.imdb_name = "coco_d"
      args.imdbval_name = "coco_2014_minival"
      args.imdb_name_org = 'coco_c'
      args.imdb_name_expert = "coco_d_expert"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "coco_b_2014":
      args.imdb_name = "coco_b_2014"
      args.imdbval_name = "coco_2014_minival"
      args.imdb_name_org = 'coco_a_2014'
      args.imdb_name_expert = "coco_b_expert_2014"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "coco_c_2014":
      args.imdb_name = "coco_c_2014"
      args.imdbval_name = "coco_2014_minival"
      args.imdb_name_org = 'coco_b_2014'
      args.imdb_name_expert = "coco_c_expert_2014"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "coco_d_2014":
      args.imdb_name = "coco_d_2014"
      args.imdbval_name = "coco_2014_minival"
      args.imdb_name_org = 'coco_c_2014'
      args.imdb_name_expert = "coco_d_expert_2014"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "tv":
      args.imdb_name = "voc_2007_1_inc_tv"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_1_train_cat'
      args.imdbval_name_org = "voc_2007_test"
      args.imdb_name_expert = "voc_2007_1_inc_tv"
  elif args.dataset == "dog":
      args.imdb_name = "voc_2007_1_inc_dog"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.imdb_name_org = 'voc_2007_1_train_cat'
      args.imdbval_name_org = "voc_2007_test"
      args.imdb_name_expert = "voc_2007_1_inc_dog"
  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  print(args.imdb_name)
  if cfg.TRAIN.IM_AUG:
      print(args.imdb_name)
      args.imdb_name=args.imdb_name+'_re'

  if args.cfg_file is not None:
    print(os.getcwd())
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  torch.manual_seed(cfg.RNG_SEED)
  np.random.seed(cfg.RNG_SEED)
  random.seed(cfg.RNG_SEED)  ##
  os.environ['PYTHONHASHSEED'] = str(cfg.RNG_SEED)
  os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
  torch.cuda.manual_seed(cfg.RNG_SEED)
  torch.cuda.manual_seed_all(cfg.RNG_SEED)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False



  #torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = True
  if 'coco' in args.dataset:
      cfg.TRAIN.USE_FLIPPED = False
  cfg.USE_GPU_NMS = args.cuda
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
  imdb_org, roidb_org, ratio_list_org, ratio_index_org = combined_roidb(args.imdb_name_org)


  # imdb_expert, roidb_expert, ratio_list_expert, ratio_index_expert = combined_roidb(args.imdb_name_expert)

  print('{:d} roidb entries'.format(len(roidb)))

  if not os.path.exists(args.save_dir):
      os.makedirs(args.save_dir)

  output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)



  dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True)
  dataset_woaug = roibatchLoader_org(roidb[:len(ratio_index)], ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True)
  train_size = len(roidb)
  if cfg.TRAIN.GRAY_AUG:
      train_size_unaug = train_size - cfg.TRAIN.GRAY_NUM
  else:
      train_size_unaug = train_size

  sampler_batch = sampler(train_size, args.batch_size)
  sampler_batch_unaug = sampler(train_size_unaug, args.batch_size)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            sampler=sampler_batch, num_workers=args.num_workers, worker_init_fn=worker_init_fn)
  dataloader_woaug = torch.utils.data.DataLoader(dataset_woaug, batch_size=args.batch_size,
                                           sampler=sampler_batch_unaug, num_workers=args.num_workers, worker_init_fn=worker_init_fn)

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.cuda:
    cfg.CUDA = True

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN_org = vgg16(imdb_org.classes, pretrained=True, class_agnostic=args.class_agnostic)
    fasterRCNN_inc = vgg16_inc(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
    basefrcnn_load_name = 'models/vgg16/pascal_voc_0712/faster_rcnn_1_20.pth'
  elif args.net == 'res101':
    fasterRCNN_org = resnet(imdb_org.classes, 101, pretrained=True, class_agnostic=args.class_agnostic,model_path='data/pretrained_model/resnet101_caffe.pth')
    fasterRCNN_inc = resnet_inc_bbox_distil(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic,model_path='data/pretrained_model/resnet101_caffe.pth')
    basefrcnn_load_name = 'models_res101_voc19/res101/pascal_voc_0712/faster_rcnn_1_20.pth'
    # fasterRCNN_residual = resnet_residual(imdb_expert.classes, 101, pretrained=True,
    #                                       class_agnostic=args.class_agnostic,model_path='data/pretrained_model/resnet101_caffe.pth')  # imdb_expert.classes
  elif args.net == 'res50':
    fasterRCNN_org = resnet(imdb_org.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
    fasterRCNN_inc = resnet_inc_bbox_distil(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
    fasterRCNN_inc_eval = resnet_split_clsbox_test(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    # fasterRCNN_residual = resnet_residual(imdb_expert.classes, 50, pretrained=True,
    #                                       class_agnostic=args.class_agnostic)  # imdb_expert.classes
    basefrcnn_load_name = 'model_save_dir/models_res50_caffe_voc07_19/res50/pascal_voc/faster_rcnn_1_20_9873.pth'#'models_res50_voc19/res50/pascal_voc_0712/faster_rcnn_1_18_32597.pth'
    if "pascal_voc_07_15" in args.dataset:
        basefrcnn_load_name = 'model_save_dir/models_res50_voc15_new/res50/pascal_voc_07_15/faster_rcnn_1_20_9003.pth'
    if "pascal_voc_07_10" in args.dataset:
        basefrcnn_load_name = 'model_save_dir/models_res50_voc10/res50/pascal_voc_07_10/faster_rcnn_1_20_6003.pth'
    if "coco" in args.dataset:
        basefrcnn_load_name = 'model_save_dir/models_res50_coco40/res50/coco_40_train/faster_rcnn_1_20_34040.pth'
  elif args.net == 'res152':
    fasterRCNN_org = resnet(imdb_org.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
    fasterRCNN_inc = resnet_inc_bbox_distil(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
    basefrcnn_load_name = 'models_res152_voc19/res152/pascal_voc_0712/faster_rcnn_1_20.pth'
  else:
    print("network is not defined")
    pdb.set_trace()

  if args.load_model!="":
    basefrcnn_load_name=args.load_model
  #expert_model=args.expert_model


  # fasterRCNN_residual.create_architecture()

  fasterRCNN_org.create_architecture()
  fasterRCNN_inc.create_architecture()
  fasterRCNN_inc_eval.create_architecture()

  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr
  #tr_momentum = cfg.TRAIN.MOMENTUM
  #tr_momentum = args.momentum

  params = []
  for key, value in dict(fasterRCNN_inc.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  # for key, value in dict(fasterRCNN_residual.named_parameters()).items():
  #     if value.requires_grad:
  #         if 'bias' in key:
  #             params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
  #                         'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
  #         else:
  #             params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  if args.cuda:
    fasterRCNN_org.cuda()
    fasterRCNN_inc.cuda()
    fasterRCNN_inc_eval.cuda()
    # fasterRCNN_residual.cuda()

  if args.optimizer == "adam":
    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

  elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

  if args.resume:
    load_name = os.path.join(output_dir,
      'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.session = checkpoint['session']
    args.start_epoch = checkpoint['epoch']
    fasterRCNN_inc.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = optimizer.param_groups[0]['lr']
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))


  print("load checkpoint %s" % (basefrcnn_load_name))
  if args.cuda > 0:
    checkpoint_org = torch.load(basefrcnn_load_name)
    #checkpoint_expert = torch.load(expert_model)
  else:
    checkpoint_org = torch.load(basefrcnn_load_name, map_location=(lambda storage, loc: storage))
    #checkpoint_expert = torch.load(expert_model, map_location=(lambda storage, loc: storage))
  '''
  checkpoint_1 = torch.load('model_save_dir/models_res50_voc10/res50/pascal_voc_07_10/faster_rcnn_1_20_6003.pth')
  checkpoint_org['model']['RCNN_bbox_pred.weight'] = torch.cat(
      (checkpoint_1['model']['RCNN_bbox_pred.weight'], checkpoint_org['model']['RCNN_bbox_pred_new.weight']), dim=0)
  checkpoint_org['model']['RCNN_cls_score.weight'] = torch.cat(
      (checkpoint_1['model']['RCNN_cls_score.weight'], checkpoint_org['model']['RCNN_cls_score_new.weight']), dim=0)
  checkpoint_org['model']['RCNN_bbox_pred.bias'] = torch.cat(
      (checkpoint_1['model']['RCNN_bbox_pred.bias'], checkpoint_org['model']['RCNN_bbox_pred_new.bias']), dim=0)
  checkpoint_org['model']['RCNN_cls_score.bias'] = torch.cat(
      (checkpoint_1['model']['RCNN_cls_score.bias'], checkpoint_org['model']['RCNN_cls_score_new.bias']), dim=0)
  '''
  if args.first > 0:
    if args.trained_residual_model:
        trained_residual_model=args.trained_residual_model
        if args.cuda > 0:
            checkpoint_tres = torch.load(trained_residual_model)
        else:
            checkpoint_tres = torch.load(trained_residual_model, map_location=(lambda storage, loc: storage))
        imdb_tres, roidb_tres, ratio_list_tres, ratio_index_tres = combined_roidb(args.imdb_name_last_expert)
        checkpoint_org['model']['RCNN_bbox_pred.weight'] = torch.cat(
            (checkpoint_org['model']['RCNN_bbox_pred.weight'], checkpoint_tres['model']['RCNN_bbox_pred.weight'][4:]),
            dim=0)
        checkpoint_org['model']['RCNN_cls_score.weight'] = torch.cat(
            (checkpoint_org['model']['RCNN_cls_score.weight'], checkpoint_tres['model']['RCNN_cls_score.weight'][1:]),
            dim=0)
        checkpoint_org['model']['RCNN_bbox_pred.bias'] = torch.cat(
            (checkpoint_org['model']['RCNN_bbox_pred.bias'], checkpoint_tres['model']['RCNN_bbox_pred.bias'][4:]), dim=0)
        checkpoint_org['model']['RCNN_cls_score.bias'] = torch.cat(
            (checkpoint_org['model']['RCNN_cls_score.bias'], checkpoint_tres['model']['RCNN_cls_score.bias'][1:]), dim=0)

    else:
        if not args.class_agnostic:
            checkpoint_org['model']['RCNN_bbox_pred.weight']=torch.cat((checkpoint_org['model']['RCNN_bbox_pred.weight'], checkpoint_org['model']['RCNN_bbox_pred_new.weight']),dim=0)
            checkpoint_org['model']['RCNN_bbox_pred.bias'] = torch.cat((checkpoint_org['model']['RCNN_bbox_pred.bias'], checkpoint_org['model']['RCNN_bbox_pred_new.bias']), dim=0)
        checkpoint_org['model']['RCNN_cls_score.weight']=torch.cat((checkpoint_org['model']['RCNN_cls_score.weight'], checkpoint_org['model']['RCNN_cls_score_new.weight']),dim=0)
        checkpoint_org['model']['RCNN_cls_score.bias'] = torch.cat((checkpoint_org['model']['RCNN_cls_score.bias'], checkpoint_org['model']['RCNN_cls_score_new.bias']), dim=0)



    ############### the parameters of base class of inc model are initialized by base model ##########################
    if args.base_model:
        base_model=args.base_model
        if args.cuda > 0:
            checkpoint_base = torch.load(base_model)
        else:
            checkpoint_base = torch.load(base_model, map_location=(lambda storage, loc: storage))
        imdb_base, roidb_base, ratio_list_base, ratio_index_base = combined_roidb(args.imdb_name_base)
        checkpoint_base['model']['RCNN_bbox_pred.weight'] = torch.cat(
            (checkpoint_base['model']['RCNN_bbox_pred.weight'], checkpoint_org['model']['RCNN_bbox_pred.weight'][len(imdb_base.classes)*4:]), dim=0)
        checkpoint_base['model']['RCNN_cls_score.weight'] = torch.cat(
            (checkpoint_base['model']['RCNN_cls_score.weight'], checkpoint_org['model']['RCNN_cls_score.weight'][len(imdb_base.classes):]), dim=0)
        checkpoint_base['model']['RCNN_bbox_pred.bias'] = torch.cat(
            (checkpoint_base['model']['RCNN_bbox_pred.bias'], checkpoint_org['model']['RCNN_bbox_pred.bias'][len(imdb_base.classes)*4:]), dim=0)
        checkpoint_base['model']['RCNN_cls_score.bias'] = torch.cat(
            (checkpoint_base['model']['RCNN_cls_score.bias'], checkpoint_org['model']['RCNN_cls_score.bias'][len(imdb_base.classes):]), dim=0)

        #fasterRCNN_base = resnet(imdb_base.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
        #fasterRCNN_base.create_architecture()
        #fasterRCNN_base.load_state_dict(checkpoint_base['model'])
        pretrained_dict_base = {k: v for k, v in
                       checkpoint_base['model'].items() if 'RCNN_cls_score' in k or 'RCNN_bbox_pred' in k}
    ###################################################################################################################
  #'''


  #args.session = checkpoint_expert['session']
  #args.start_epoch = checkpoint_expert['epoch']
  #fasterRCNN_residual.load_state_dict(checkpoint_residual['model'])
  #optimizer.load_state_dict(checkpoint_expert['optimizer'])
  #lr = optimizer.param_groups[0]['lr']
  #if 'pooling_mode' in checkpoint_expert.keys():
  #    cfg.POOLING_MODE = checkpoint_expert['pooling_mode']



  pretrained_dic_org = {k: v for k, v in checkpoint_org['model'].items() if 'RCNN_cls_score_new' not in k and 'RCNN_bbox_pred_new' not in k and 'discriminator' not in k }#and 'extra_fea_pool' not in k and 'extra_classifier' not in k
  #pretrained_dic_org=checkpoint_org['model']
  fasterRCNN_org.load_state_dict(pretrained_dic_org)

  #fasterRCNN_org.load_state_dict(pretrained_dic_org)
  #fasterRCNN_org.load_state_dict(checkpoint_org)

  frcnn_inc_model_dict=fasterRCNN_inc.state_dict()
  if args.first > 0 and cfg.TRAIN.isda:
      checkpoint_org['model']['extra_classifier.weight'] = torch.cat((checkpoint_org['model'][
                                                                          'extra_classifier.weight'],
                                                                      frcnn_inc_model_dict[
                                                                          'extra_classifier.weight'][
                                                                      checkpoint_org['model'][
                                                                          'extra_classifier.weight'].shape[0]:]), dim=0)
      checkpoint_org['model']['extra_classifier.bias'] = torch.cat(
          (checkpoint_org['model']['extra_classifier.bias'],
           frcnn_inc_model_dict['extra_classifier.bias'][checkpoint_org['model'][
                                                             'extra_classifier.bias'].shape[0]:]),
          dim=0)
  #pretrained_dict = {k: v for k, v in checkpoint_org['model'].items() if 'cls' not in k and 'bbox' not in k}
  #pretrained_dict = {k: v for k, v in checkpoint_org['model'].items() if 'bbox' not in k}  ################## split old and new cls
  if cfg.TRAIN.excls:
      excls_dict = { k.replace('RCNN_top', 'extra_fea_pool'): v for k, v in checkpoint_org['model'].items() if 'RCNN_top' in k}
      # print(excls_dict)
      # frcnn_inc_model_dict.update(excls_dict)
  pretrained_dict = {k: v for k, v in checkpoint_org['model'].items()}  # if 'extra_classifier' not in k if 'channel_att' not in k################# split old and new cls and bbox
  '''
  pretrained_dict_expert_cls = {k.split('.')[0]+'_new.'+k.split('.')[1]: v[1:] for k, v in checkpoint_expert['model'].items() if 'RCNN_cls' in k }#or 'RCNN_bbox' in k
  pretrained_dict_expert_box = {k.split('.')[0] + '_new.' + k.split('.')[1]: v[4:] for k, v in
                            checkpoint_expert['model'].items() if 'RCNN_bbox' in k}
                            '''
  frcnn_inc_model_dict.update(pretrained_dict)
  #frcnn_inc_model_dict.update(pretrained_dict_expert_cls) ################################### expert new initialize !!!!!!!!!!!!!!!!!!!!!!!!!!
  #frcnn_inc_model_dict.update(pretrained_dict_expert_box) ################################### expert new initialize !!!!!!!!!!!!!!!!!!!!!!!!!!
  if args.first and args.base_model:
      frcnn_inc_model_dict.update(pretrained_dict_base)
  fasterRCNN_inc.load_state_dict(frcnn_inc_model_dict)

  frcnn_inc_model_dict_eval = fasterRCNN_inc_eval.state_dict()
  frcnn_inc_model_dict_eval.update(frcnn_inc_model_dict)
  fasterRCNN_inc_eval.load_state_dict(frcnn_inc_model_dict_eval)
  ##################### base feature extraction frozen ####################
  '''
  for k, v in fasterRCNN_inc.named_parameters():
      if 'base' in k :
          print(k)
          v.requires_grad = False  #
  
      #if 'base' in k and 'RCNN_base.5' not in k and 'RCNN_base.6' not in k:
      #    print(k)
      #    v.requires_grad = False  # freeze

  #for i in fasterRCNN_inc.named_parameters():
  #    print(i)
  '''

  # for k, v in fasterRCNN_inc.named_parameters():
  #     if 'RCNN_cls_score.' in k or 'RCNN_bbox_pred.' in k:
  #         print(k)
  #         v.requires_grad = False  # freeze

  ########################### freeze !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  # for k, v in fasterRCNN_inc.named_parameters():
  #     if 'RCNN_bbox_pred' in k and 'new' not in k :#or 'RCNN_bbox_pred' in k:
  #         print(k)
  #         v.requires_grad = False  # freeze
  #     if 'RCNN_cls_score' in k and 'new' not in k:
  #         print(k)
  #         v.requires_grad = False  # freeze
  ##########################################################################

  if args.mGPUs:
    fasterRCNN_org = nn.DataParallel(fasterRCNN_org)
    fasterRCNN_inc = nn.DataParallel(fasterRCNN_inc)
    fasterRCNN_inc_eval = nn.DataParallel(fasterRCNN_inc_eval)
    # fasterRCNN_residual = nn.DataParallel(fasterRCNN_residual)

  iters_per_epoch = int(train_size / args.batch_size)

  if args.use_tfboard:
    from tensorboardX import SummaryWriter
    logger = SummaryWriter("logs")

  fasterRCNN_org.eval()
  fasterRCNN_inc_eval.eval()
  #for i in fasterRCNN_org.named_parameters():
  #    print(i)
  for k,v in fasterRCNN_org.named_parameters():
      v.requires_grad = False
  for k,v in fasterRCNN_inc_eval.named_parameters():
      v.requires_grad = False
  #channel_att_old = torch.from_numpy(np.load('c_att_flag.npz')['arr_0'])
  limit_epoch = 0


  prototypes = None
  stds = None
  prototype_label = None

  class crd_atr(object):
      def __init__(self):
          self.s_dim = 0
          self.t_dim = 0  # the dimension of teacher's feature
          self.n_data = 0  # the number of samples in the training set, therefor the memory buffer is: opt.n_data x opt.feat_dim
          self.feat_dim = 0  # the dimension of the projection space
          self.nce_k = 1  # 180#16384 # number of negatives paired with each positive
          self.nce_t = 0.07  # temperature parameter for softmax
          self.nce_m = 0.5  # momentum for non-parametric updates
  from model.crd.criterion import CRDLoss

  opt = crd_atr()
  opt.s_dim = 2048#pooled_feat.shape[1]
  opt.t_dim = 2048#pooled_feat_org.shape[1]
  opt.n_data = len(dataset)*256#pooled_feat.shape[0] * 200
  opt.feat_dim = 128  # 256#pooled_feat.shape[1]
  opt.nce_k = 160##10
  # crd_criterion = CRDLoss(opt).cuda()
  crd_criterion = None

  if cfg.TRAIN.rdc:
      feature_num = int(cfg.TRAIN.feadim / 4)
  else:
      feature_num = cfg.TRAIN.feadim
  #################################### old feature mean #################################
  preds = []
  true_labels = []
  cls_num = torch.zeros(imdb.num_classes).cuda()
  new_feamean = torch.zeros(imdb.num_classes, feature_num).cuda()

  croplist = []
  cls_list = imdb.classes#imdb_org.classes  # [1:]
  crop_path = cfg.TRAIN.CROP_PATH
  if isinstance(crop_path, list):
      crop_path_all = crop_path + [cfg.TRAIN.NEW_PATH]
  else:
      crop_path_all = [crop_path] + [cfg.TRAIN.NEW_PATH]
  if isinstance(crop_path_all, list):
      for crp in crop_path_all:
          getFile(crp, croplist)
  else:
      getFile(crop_path_all, croplist)
  crop_feas_old = torch.zeros(len(imdb_org.classes) - 1, feature_num).cuda()
  crop_prototypes = torch.zeros(len(imdb_org.classes) - 1, 2048).cuda()
  if cfg.TRAIN.CROP_AUG:# or not cfg.TRAIN.IM_AUG:
      for idx_crop, _ in enumerate(croplist):
          im_file = os.path.join('', croplist[idx_crop])
          cls_idx = float(cls_list.index(croplist[idx_crop].split('/')[-2]))
          # im = cv2.imread(im_file)
          im_in = cv2.imread(im_file)[:, :, ::-1]
          # im_in = np.array(imread(im_file))
          if len(im_in.shape) == 2:
              im_in = im_in[:, :, np.newaxis]
              im_in = np.concatenate((im_in, im_in, im_in), axis=2)
          # rgb -> bgr
          im = im_in[:, :, ::-1]

          blobs, im_scales = _get_image_blob(im)
          assert len(im_scales) == 1, "Only single-image batch implemented"
          im_blob = blobs
          im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
          im_data_pt = torch.from_numpy(im_blob)
          im_data_pt = im_data_pt.permute(0, 3, 1, 2)
          im_info_pt = torch.from_numpy(im_info_np)
          with torch.no_grad():
              im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
              im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
              # gt_boxes.resize_(1, 1, 5).zero_()
              gt_boxes = torch.Tensor([[[0, 0, im.shape[1], im.shape[0], cls_idx]]]).cuda()
              gt_boxes[0, 0, 0:4] *= im_scales[0]
              num_boxes = torch.Tensor([1]).cuda().long()  # .resize_(1).zero_()
          # rois_crop, cls_prob_crop, bbox_pred_crop, \
          # rpn_loss_cls_crop, rpn_loss_box_crop, \
          # RCNN_loss_cls_crop, RCNN_loss_bbox_crop, \
          # rois_label_crop, _ = fasterRCNN_org(im_data, im_info, gt_boxes, num_boxes, extract_gtfea=True, isda=True)
          crop_fea, crop_label, crop_fea_2048, crop_pred_prob = fasterRCNN_org(im_data, im_info, gt_boxes, num_boxes, extract_gtfea=True,
                                                               isda=True,
                                                               draw=False)
          if cls_idx<imdb_org.num_classes:
              # crop_fea = crop_fea.mean(3).mean(2)
              crop_feas_old[int(cls_idx) - 1] = crop_fea
              crop_prototypes[int(cls_idx) - 1] = crop_fea_2048
          else:
              pred = torch.argmax(crop_pred_prob[0], dim=0)
              if pred>0:
                  preds.append(pred)
                  true_labels.append(crop_label)
                  cls_num[crop_label] += 1
                  new_feamean[crop_label] += crop_fea
  ########################################################################################


  if cfg.TRAIN.isda:
      ######################## isda loss #####################################################
      # isda_criterion = ISDALoss(int(model.feature_num), class_num).cuda()
      # loss, output = isda_criterion(model, fc, input_var, target_var, ratio)

      from model.faster_rcnn.ISDA import EstimatorCV, ISDALoss
      # from model.faster_rcnn.ISDA_dsl import EstimatorCV, ISDALoss

      class_num = imdb.num_classes-1#imdb_org.num_classes-1# #imdb.num_classes# - 1imdb.num_classes - 1#
      isda_criterion = ISDALoss(feature_num, class_num, old_num = imdb_org.num_classes-1).cuda()
      # ratio = 0.5  * (epoch / (training_configurations[args.model]['epochs']))
      # isda_loss, isda_output = isda_criterion(fasterRCNN_org, fasterRCNN_org.RCNN_cls_score, pooled_feat, rois_label, ratio)
      ###########################################################################################
  else:
      isda_criterion = None

  if cfg.TRAIN.isda or cfg.TRAIN.kg:
      # from sklearn.metrics import confusion_matrix
      def confusion_matrix(preds, labels, conf_matrix):
          for p, t in zip(preds, labels):
              conf_matrix[p, t] += 1
          return conf_matrix

      '''
      preds = []
      true_labels = []
      cls_num = torch.zeros(imdb.num_classes).cuda()
      new_feamean = torch.zeros(imdb.num_classes, feature_num).cuda()
      #################################### isda confusion matrix ####################################
      data_iter = iter(dataloader_woaug)
      for step in range(int(train_size_unaug / args.batch_size)):
          data = next(data_iter)
          if data == None:
              print(data)
          with torch.no_grad():
              im_data.resize_(data[0].size()).copy_(data[0])
              im_info.resize_(data[1].size()).copy_(data[1])
              gt_boxes.resize_(data[2].size()).copy_(data[2])
              num_boxes.resize_(data[3].size()).copy_(data[3])
          #################frcnn_org_eval######################################
          rois_org, cls_prob_org, bbox_pred_org, \
          rpn_loss_cls_org, rpn_loss_box_org, \
          RCNN_loss_cls_org, RCNN_loss_bbox_org, \
          rois_label_org, rois_fea_1024 = fasterRCNN_org(im_data, im_info, gt_boxes, num_boxes, extract_gtfea=True)
          scores_org = cls_prob_org.data
          boxes_org = rois_org.data[:, :, 1:5]
          pred = torch.argmax(cls_prob_org[0][:rois_label_org.shape[0]], dim=1)
          nonbg = (pred > 0).nonzero().view(-1)
          pred = pred[nonbg]
          if len(pred) > 0:
              for pr, rl, feat_1024 in zip(pred, rois_label_org[nonbg], rois_fea_1024[nonbg]):
                  preds.append(pr)
                  true_labels.append(rl)
                  cls_num[rl] += 1
                  new_feamean[rl] += feat_1024
      '''
      crop_feas_new = new_feamean[imdb_org.num_classes:].div(cls_num[imdb_org.num_classes:].unsqueeze(dim=1))
      preds = torch.stack(preds).cpu().numpy()
      true_labels = torch.stack(true_labels).cpu().numpy()
      # cf = confusion_matrix(true_labels, preds).astype(float).t()  # 69,69
      cf = np.zeros((imdb.num_classes - 1, imdb.num_classes - 1))
      confusion_matrix(preds - 1, true_labels - 1, cf)
      cf = cf.transpose()
      cf_normalized = cf.astype('float') / cf.sum(axis=1)[:, np.newaxis]
      cf_normalized = np.round(cf_normalized, 2)
      kg = torch.from_numpy(cf_normalized).cuda()  # [imdb_org.num_classes-1:]

      crop_feas = torch.cat((crop_feas_old, crop_feas_new), dim=0)
      # use kg to get reasoning prototype

      out_new = (crop_feas[:imdb_org.num_classes - 1] + torch.matmul(
          kg[imdb_org.num_classes - 1:, :imdb_org.num_classes - 1].float().t(),
          crop_feas[imdb_org.num_classes - 1:])) / 2  # new class fea merge for old class
      out_new = torch.cat((out_new, crop_feas[imdb_org.num_classes - 1:]), dim=0)
      out_new = out_new - crop_feas
      isda_tmp = (kg, out_new, crop_feas)
      np.save('kg.npy', kg.cpu())
  else:
      isda_tmp = None

  time_0 = time.time()
  croplist_train = []
  cls_list_train = imdb.classes  # [1:]
  if isinstance(cfg.TRAIN.CROP_PATH, list):
      crop_path_train = copy.deepcopy(cfg.TRAIN.CROP_PATH)
      crop_path_train.append(cfg.TRAIN.NEW_PATH)
  else:
      crop_path_train = [cfg.TRAIN.CROP_PATH, cfg.TRAIN.NEW_PATH]
  print('crops list: {}'.format(crop_path_train))
  print('cfg.TRAIN.CROP_PATH', cfg.TRAIN.CROP_PATH, 'crop_path_train', crop_path_train)
  if isinstance(crop_path_train, list):
      for crp in crop_path_train:
          getFile(crp, croplist_train)
  else:
      getFile(crop_path_train, croplist_train)
  croplist_train_old = croplist_train[:imdb_org.num_classes - 1]
  croplist_train_new = croplist_train[imdb_org.num_classes - 1:]


  pretrain_ex = False#True#
  if pretrain_ex and cfg.TRAIN.isda:
      for epoch in range(args.start_epoch, int((args.max_epochs)/2)+ 1 ):
          fasterRCNN_inc.train()
          if epoch<0:#<0:#(args.max_epochs)*2+1:
              # num_cls = torch.zeros(imdb.num_classes)
              # for idx_crop in range(len(croplist_train)):
              #     cls_idx = float(cls_list_train.index(croplist_train[idx_crop].split('/')[-2]))
              #     num_cls[cls_idx]+=1
              for idx_crop in range(len(croplist_train)):
                  crop_batch = []
                  clss = []
                  ims = []
                  cbs = random.choice([1,4,6])#,9])
                  cbs = random.choice([1])#,9])

                  csize = random.choice([64, 96, 128])
                  bg_img = superpixel(image=cv2.imread(random.choice(croplist_train_new))[:, :, ::-1])
                  if cbs == 1:
                      im_file = random.choice(croplist_train)  # os.path.join('', croplist_train[idx_crop])
                      cls_idx = float(cls_list_train.index(im_file.split('/')[-2]))
                      im_in = cv2.imread(im_file)[:, :, ::-1]
                      im_in = seq(image=im_in)  ######### augmentation
                      gt_boxes = [[0, 0, im_in.shape[1], im_in.shape[0], cls_idx]]
                  else:
                      for cb in range(cbs):
                          if cb < int(cbs/2):
                              im_file = random.choice(croplist_train_old)  # os.path.join('', croplist_train[idx_crop])
                          else:
                              im_file = random.choice(croplist_train_new)  # os.path.join('', croplist_train[idx_crop])
                          cls_idx = float(cls_list_train.index(im_file.split('/')[-2]))
                          im_in = cv2.resize(cv2.imread(im_file),(csize,csize))[:, :, ::-1]
                          im_in = seq(image=im_in) ######### augmentation
                          ims.append(im_in)
                          clss.append(cls_idx)
                      if cbs == 4:
                          im_in = merge_image(ims, 2, 2, csize)
                          gt_boxes = [[0*csize, 0*csize, 1*csize, 1*csize, clss[0]], [1*csize, 0*csize, 2*csize, 1*csize, clss[1]],
                                      [0*csize, 1*csize, 1*csize, 2*csize, clss[2]], [1*csize, 1*csize, 2*csize, 2*csize, clss[3]]]
                      elif cbs == 6:
                          im_in = merge_image(ims, 2, 3, csize)
                          gt_boxes = [[0*csize, 0*csize, 1*csize, 1*csize, clss[0]], [1*csize, 0*csize, 2*csize, 1*csize, clss[1]], [2*csize, 0,     3*csize,   csize, clss[2]],
                                      [0*csize, 1*csize, 1*csize, 2*csize, clss[3]], [1*csize, 1*csize, 2*csize, 2*csize, clss[4]], [2*csize, csize, 3*csize, 2*csize, clss[5]]]
                      elif cbs == 9:
                          im_in = merge_image(ims, 3, 3, csize)
                          gt_boxes = [[0*csize, 0*csize, 1*csize, 1*csize, clss[0]], [1*csize, 0*csize, 2*csize, 1*csize, clss[1]], [2*csize, 0*csize, 3*csize, 1*csize, clss[2]],
                                      [0*csize, 1*csize, 1*csize, 2*csize, clss[3]], [1*csize, 1*csize, 2*csize, 2*csize, clss[4]], [2*csize, 1*csize, 3*csize, 2*csize, clss[5]],
                                      [0*csize, 2*csize, 1*csize, 3*csize, clss[6]], [1*csize, 2*csize, 2*csize, 3*csize, clss[7]], [2*csize, 2*csize, 3*csize, 3*csize, clss[8]]]
                  bg_img = cv2.resize(bg_img, (im_in.shape[1]*2, im_in.shape[0]*2))
                  tx, ty = random.randint(0, im_in.shape[1]), random.randint(0, im_in.shape[0])
                  bg_img[ty:ty+im_in.shape[0], tx:tx+im_in.shape[1]] = im_in
                  im_in = bg_img
                  gt_boxes = np.array(gt_boxes)
                  gt_boxes[:, 0] += tx
                  gt_boxes[:, 2] += tx
                  gt_boxes[:, 1] += ty
                  gt_boxes[:, 3] += ty

                  if len(im_in.shape) == 2:
                      im_in = im_in[:, :, np.newaxis]
                      im_in = np.concatenate((im_in, im_in, im_in), axis=2)
                  # rgb -> bgr
                  im = im_in[:, :, ::-1]

                  blobs, im_scales = _get_image_blob(im)
                  assert len(im_scales) == 1, "Only single-image batch implemented"
                  im_blob = blobs
                  im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
                  im_data_pt = torch.from_numpy(im_blob)
                  im_data_pt = im_data_pt.permute(0, 3, 1, 2)
                  im_info_pt = torch.from_numpy(im_info_np)
                  with torch.no_grad():
                      im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
                      im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
                      # gt_boxes.resize_(1, 1, 5).zero_()
                      gt_boxes = torch.Tensor([gt_boxes]).cuda()#torch.Tensor([[[0, 0, im.shape[1], im.shape[0], cls_idx]]]).cuda()
                      gt_boxes[0, :, 0:4] *= im_scales[0]
                      num_boxes = torch.Tensor([4]).cuda().long()  # .resize_(1).zero_()

                  fasterRCNN_inc.zero_grad()
                  # if epoch>limit_epoch:
                  rois, cls_prob, bbox_pred, \
                  rpn_loss_cls, rpn_loss_box, \
                  RCNN_loss_cls, RCNN_loss_bbox, \
                  rois_label, rcnn_cls_distil_loss, rcnn_bbox_distil_loss, base_feat_distil_loss, \
                  rpn_conv1_distil_loss, pooled_feat_distil_loss, cos_loss, c_att, isda_loss, proto_loss \
                      = fasterRCNN_inc(im_data, im_info, gt_boxes, num_boxes, fasterRCNN_org=fasterRCNN_org, step=idx_crop, roidb=dataset._roidb, \
                                       ratio_index=dataloader.batch_sampler.sampler.rand_num_view, prototypes=prototypes,
                                       stds=stds, fasterRCNN_residual=None, channel_att_old=None, se_flag=True,
                                       crd_criterion=crd_criterion, opt=opt, prototype_label=prototype_label,
                                       isda_criterion=isda_criterion, epoch=epoch, isda_tmp=isda_tmp)
                  # loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                  #        + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean() \
                  #        + rcnn_cls_distil_loss.mean() + rcnn_bbox_distil_loss.mean() + base_feat_distil_loss.mean()  # + rpn_embed_distil_loss.mean() #+ margin_loss.mean()  ############### distil#+ rpn_cls_distil_loss.mean() + rpn_bbox_distil_loss.mean() \
                  # loss += pooled_feat_distil_loss.mean() + lmd * (rpn_conv1_distil_loss.mean() + cos_loss.mean())
                  loss = isda_loss.mean()
                  # backward
                  optimizer.zero_grad()
                  loss.backward()
                  optimizer.step()
                  if idx_crop % 100 == 0:
                      time_1 = time.time()
                      print(
                          '*** [ epoch  %3d  ], [ step  %3d  ], [time %2.4f ], ************************** extra classifier loss: %.4f' % (
                          epoch, idx_crop, time_1 - time_0, loss.item()))
                      time_0 = time_1
          else:
              data_iter = iter(dataloader)
              for step in range(iters_per_epoch):
                  # for i in fasterRCNN_org.named_parameters():
                  #    print('train:', i)
                  data = next(data_iter)
                  if data == None:
                      print(data)
                  with torch.no_grad():
                      im_data.resize_(data[0].size()).copy_(data[0])
                      im_info.resize_(data[1].size()).copy_(data[1])
                      gt_boxes.resize_(data[2].size()).copy_(data[2])
                      num_boxes.resize_(data[3].size()).copy_(data[3])
                  #################frcnn_org_eval######################################
                  rois_org, cls_prob_org, bbox_pred_org, \
                  rpn_loss_cls_org, rpn_loss_box_org, \
                  RCNN_loss_cls_org, RCNN_loss_bbox_org, \
                  rois_label_org, _ = fasterRCNN_org(im_data, im_info, gt_boxes, num_boxes)
                  scores_org = cls_prob_org.data
                  boxes_org = rois_org.data[:, :, 1:5]

                  fasterRCNN_inc.zero_grad()

                  rois, cls_prob, bbox_pred, \
                  rpn_loss_cls, rpn_loss_box, \
                  RCNN_loss_cls, RCNN_loss_bbox, \
                  rois_label, rcnn_cls_distil_loss, rcnn_bbox_distil_loss, base_feat_distil_loss, \
                  rpn_conv1_distil_loss, pooled_feat_distil_loss, cos_loss, c_att, isda_loss, mixup_loss, proto_loss \
                      = fasterRCNN_inc(im_data, im_info, gt_boxes, num_boxes, rois_org, cls_prob_org, bbox_pred_org,
                                       rois_label_org, fasterRCNN_org, step, dataset._roidb, \
                                       ratio_index=dataloader.batch_sampler.sampler.rand_num_view,
                                       prototypes=prototypes, stds=stds, fasterRCNN_residual=None, channel_att_old=None,
                                       se_flag=True, crd_criterion=crd_criterion, opt=opt,
                                       prototype_label=prototype_label, isda_criterion=isda_criterion, epoch=epoch,
                                       isda_tmp=isda_tmp)
                  # if epoch>limit_epoch:
                  # rois, cls_prob, bbox_pred, \
                  # rpn_loss_cls, rpn_loss_box, \
                  # RCNN_loss_cls, RCNN_loss_bbox, \
                  # rois_label, rcnn_cls_distil_loss, rcnn_bbox_distil_loss, base_feat_distil_loss, \
                  # rpn_conv1_distil_loss, pooled_feat_distil_loss, cos_loss, c_att, isda_loss \
                  #     = fasterRCNN_inc(im_data, im_info, gt_boxes, num_boxes, fasterRCNN_org=fasterRCNN_org, step=step,
                  #                      roidb=dataset._roidb, \
                  #                      ratio_index=dataloader.batch_sampler.sampler.rand_num_view, prototypes=prototypes,
                  #                      stds=stds, fasterRCNN_residual=None, channel_att_old=None, se_flag=True,
                  #                      crd_criterion=crd_criterion, opt=opt, prototype_label=prototype_label,
                  #                      isda_criterion=isda_criterion, epoch=epoch, isda_tmp=isda_tmp)


                  # loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                  #        + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean() \
                  #        + rcnn_cls_distil_loss.mean() + rcnn_bbox_distil_loss.mean() + base_feat_distil_loss.mean()  # + rpn_embed_distil_loss.mean() #+ margin_loss.mean()  ############### distil#+ rpn_cls_distil_loss.mean() + rpn_bbox_distil_loss.mean() \
                  # loss += pooled_feat_distil_loss.mean() + lmd * (rpn_conv1_distil_loss.mean() + cos_loss.mean())
                  loss = isda_loss.mean()
                  # backward
                  optimizer.zero_grad()
                  loss.backward()
                  optimizer.step()
                  if step%100==0:
                      time_1 = time.time()
                      print('*** [ epoch  %d  ], [ step  %d  ], [time %.4f ], ************************** extra classifier loss: %.4f' % (epoch, step, time_1-time_0, loss.item()))
                      time_0 = time_1


      save_name = os.path.join(output_dir, 'faster_rcnn_ex.pth')
      save_checkpoint({
          'session': args.session,
          'epoch': epoch,
          'model': fasterRCNN_inc.module.state_dict() if args.mGPUs else fasterRCNN_inc.state_dict(),
          'optimizer': optimizer.state_dict(),
          'pooling_mode': cfg.POOLING_MODE,
          'class_agnostic': args.class_agnostic,
      }, save_name)
      print('save model: {}'.format(save_name))
      # args.max_epochs = 0

      for k, v in fasterRCNN_inc.named_parameters():
          if 'extra' in k:  # or 'RCNN_bbox_pred' in k:
              print(k)
              v.requires_grad = False  # freeze


  for epoch in range(args.start_epoch, args.max_epochs + 1):
    # setting to train mode

    fasterRCNN_inc.train()
    # fasterRCNN_residual.train()
    # if epoch>limit_epoch:
    #     for k, v in fasterRCNN_inc.named_parameters():
    #         if 'channel_att' not in k:  # or 'RCNN_bbox_pred' in k:
    #             v.requires_grad = False  # freeze
    loss_temp = 0
    start = time.time()

    if epoch % (args.lr_decay_step + 1) == 0:
        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        lr *= args.lr_decay_gamma

    torch.autograd.set_detect_anomaly(True)

    data_iter = iter(dataloader)
    for step in range(iters_per_epoch):

      #for i in fasterRCNN_org.named_parameters():
      #    print('train:', i)
      data = next(data_iter)
      if data==None:
          print(data)
      with torch.no_grad():
          im_data.resize_(data[0].size()).copy_(data[0])
          im_info.resize_(data[1].size()).copy_(data[1])
          gt_boxes.resize_(data[2].size()).copy_(data[2])
          num_boxes.resize_(data[3].size()).copy_(data[3])
          gray_flag = data[4]
          if len(data[-1].shape)>2:
              # gt_boxes = gt_boxes[:,(gt_boxes[:,:,-1]!=-2).nonzero()]
              # num_boxes = num_boxes-((gt_boxes[:, :, -1] != -2).nonzero()).shape[0]
              bg_coords = data[-1]#[:, :, 0:4]*torch.from_numpy(im_scales[:])#.unsqueeze(dim=0)
          else:
              bg_coords = None
      if len(gt_boxes[0][:num_boxes[0]].nonzero().view(-1))==0:
          continue
      #################frcnn_org_eval######################################
      rois_org, cls_prob_org, bbox_pred_org, \
      rpn_loss_cls_org, rpn_loss_box_org, \
      RCNN_loss_cls_org, RCNN_loss_bbox_org, \
      rois_label_org,_ = fasterRCNN_org(im_data, im_info, gt_boxes, num_boxes)
      scores_org = cls_prob_org.data
      boxes_org = rois_org.data[:, :, 1:5]
      #####################################################################
      if cfg.TRAIN.PSE_NEW and epoch>cfg.TRAIN.pn_ratio and gray_flag:#0:# 0:#epoch >int((args.max_epochs+1)/cfg.TRAIN.pn_ratio)
          frcnn_inc_model_dict = fasterRCNN_inc.state_dict()
          frcnn_inc_model_dict_eval.update(frcnn_inc_model_dict)
          fasterRCNN_inc_eval.load_state_dict(frcnn_inc_model_dict_eval)
          rois_inc, cls_prob_inc, bbox_pred_inc, \
          rpn_loss_cls_inc, rpn_loss_box_inc, \
          RCNN_loss_cls_inc, RCNN_loss_bbox_inc, \
          rois_label_inc, _ = fasterRCNN_inc_eval(im_data, im_info, gt_boxes, num_boxes)
          scores_inc = cls_prob_inc.data
          boxes_inc = rois_inc.data[:, :, 1:5]
      else:
          cls_prob_inc, bbox_pred_inc, rois_inc = None, None, None
      fasterRCNN_inc.zero_grad()
      # fasterRCNN_residual.zero_grad()



      #if epoch>limit_epoch:
      rois, cls_prob, bbox_pred, \
          rpn_loss_cls, rpn_loss_box, \
          RCNN_loss_cls, RCNN_loss_bbox, \
          rois_label,rcnn_cls_distil_loss,rcnn_bbox_distil_loss,base_feat_distil_loss, \
          rpn_conv1_distil_loss, pooled_feat_distil_loss, cos_loss, c_att, isda_loss, mixup_loss, proto_loss \
              = fasterRCNN_inc(im_data, im_info, gt_boxes, num_boxes, rois_org,cls_prob_org,bbox_pred_org, rois_label_org, fasterRCNN_org, step, dataset._roidb, \
                               ratio_index=dataloader.batch_sampler.sampler.rand_num_view, prototypes=crop_prototypes, stds = stds, fasterRCNN_residual=None, channel_att_old=None,se_flag=True,crd_criterion=crd_criterion, \
                               opt=opt, prototype_label=prototype_label, isda_criterion=isda_criterion, epoch=epoch, \
                               isda_tmp=isda_tmp, exsup=True, gray_flag=gray_flag, bg_coords=bg_coords, rois_inc = rois_inc, cls_prob_inc=cls_prob_inc, bbox_pred_inc=bbox_pred_inc)
          #margin_loss,rpn_cls_distil_loss, rpn_bbox_distil_loss \      rpn_embed_distil_loss \
      # else:
      #     rois, cls_prob, bbox_pred, \
      #     rpn_loss_cls, rpn_loss_box, \
      #     RCNN_loss_cls, RCNN_loss_bbox, \
      #     rois_label, rcnn_cls_distil_loss, rcnn_bbox_distil_loss, base_feat_distil_loss, \
      #     rpn_conv1_distil_loss, pooled_feat_distil_loss, cos_loss, c_att \
      #         = fasterRCNN_inc(im_data, im_info, gt_boxes, num_boxes, rois_org, cls_prob_org, bbox_pred_org,
      #                          rois_label_org, fasterRCNN_org, step, dataset._roidb, ratio_index,
      #                          fasterRCNN_residual=None, channel_att_old=None)
      #     # margin_loss,rpn_cls_distil_loss, rpn_bbox_distil_loss \      rpn_embed_distil_loss \
      '''
      fasterRCNN_residual.zero_grad()
      rois_r, cls_prob_r, bbox_pred_r, \
      rpn_loss_cls_r, rpn_loss_box_r, \
      RCNN_loss_cls_r, RCNN_loss_bbox_r, \
      rois_label_r,base_feat_distil_loss_r \
          = fasterRCNN_residual(im_data, im_info, gt_boxes, num_boxes, base_feat_residual, rcnn_cls_distil_residual, rcnn_bbox_distil_residual)
      '''
      lmd=1#2#10#0.1#1
      loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
           + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean() \
           + rcnn_cls_distil_loss.mean() + rcnn_bbox_distil_loss.mean() + base_feat_distil_loss.mean() #+ rpn_embed_distil_loss.mean() #+ margin_loss.mean()  ############### distil#+ rpn_cls_distil_loss.mean() + rpn_bbox_distil_loss.mean() \
      loss += pooled_feat_distil_loss.mean()+lmd*(rpn_conv1_distil_loss.mean()+cos_loss.mean())
      loss += isda_loss.mean()
      loss += mixup_loss.mean()
      loss += proto_loss.mean()

      # loss += rpn_loss_cls_r.mean() + rpn_loss_bbox_r.mean() \
      #      + RCNN_loss_cls_r.mean() + RCNN_loss_bbox_r.mean() \
      #      + base_feat_residual_loss.mean() + pooled_feat_distil_loss.mean()

      loss_temp += loss.item()

      # backward
      optimizer.zero_grad()
      try:
        loss.backward()
      except Exception as e:
        print(loss, e)
      if args.net == "vgg16":
          clip_gradient(fasterRCNN_inc, 10.)
      optimizer.step()

      if step % args.disp_interval == 0:
        end = time.time()
        if step > 0:
          loss_temp /= (args.disp_interval + 1)

        if args.mGPUs:
          loss_rpn_cls = rpn_loss_cls.mean().item()
          loss_rpn_box = rpn_loss_box.mean().item()
          loss_rcnn_cls = RCNN_loss_cls.mean().item()
          loss_rcnn_box = RCNN_loss_bbox.mean().item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt
          loss_rcnn_cls_distil_loss=rcnn_cls_distil_loss.mean.item()
          loss_rcnn_bbox_distil_loss = rcnn_bbox_distil_loss.mean.item()
          loss_base_feat_distil_loss = base_feat_distil_loss.mean.item()
          #loss_margin_loss = margin_loss.mean.item()
          #loss_rpn_cls_distil_loss = rpn_cls_distil_loss.mean.item()
          #loss_rpn_bbox_distil_loss = rpn_bbox_distil_loss.mean.item()
        else:
          loss_rpn_cls = rpn_loss_cls.item()
          loss_rpn_box = rpn_loss_box.item()
          loss_rcnn_cls = RCNN_loss_cls.item()
          loss_rcnn_box = RCNN_loss_bbox.item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt
          loss_rcnn_cls_distil_loss = rcnn_cls_distil_loss.item()       ############# distil
          loss_rcnn_bbox_distil_loss = rcnn_bbox_distil_loss.item()
          loss_base_feat_distil_loss = base_feat_distil_loss.item()

          # loss_rpn_cls_r = rpn_loss_cls_r.item()
          # loss_rpn_box_r = rpn_loss_bbox_r.item()
          # loss_rcnn_cls_r = RCNN_loss_cls_r.item()
          # loss_rcnn_box_r = RCNN_loss_bbox_r.item()
          # loss_base_feat_residual = base_feat_residual_loss.item()

          loss_rpn_conv1_distil_loss = rpn_conv1_distil_loss.item()
          loss_pooled_feat_distil_loss = pooled_feat_distil_loss.item()
          loss_cos_loss = cos_loss.item()
          #loss_margin_loss = 0#margin_loss.item()
          #loss_rpn_cls_distil_loss = rpn_cls_distil_loss.item()
          #loss_rpn_bbox_distil_loss = rpn_bbox_distil_loss.item()
          #loss_rcnn_cls_distil_loss=0####################
          loss_isda = isda_loss.item()
          loss_mixup = mixup_loss.item()
          loss_proto = proto_loss.item()



        print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
        print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
        print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f, rcnn_cls_distil_loss %.8f, rcnn_bbox_distil_loss %.8f, base_feat_distil_loss %.4f, rpn_conv1_distil_loss %.4f, pooled_feat_distil_loss %.4f, cos_loss %.8f, isda_loss %.4f, mixup_loss %.4f, proto_loss %.4f" \
        % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box,\
            loss_rcnn_cls_distil_loss,rcnn_bbox_distil_loss,base_feat_distil_loss, \
           loss_rpn_conv1_distil_loss, loss_pooled_feat_distil_loss, loss_cos_loss, loss_isda, loss_mixup, loss_proto \
                         ) )# , margin_loss %.4f, rpn_cls_distil_loss %.4f, rpn_bbox_distil_loss %.4f#,loss_margin_loss,loss_rpn_cls_distil_loss,loss_rpn_bbox_distil_loss rpn_embed_distil_loss : %.8f rpn_embed_distil_loss.item()
        if args.use_tfboard:
          info = {
            'loss': loss_temp,
            'loss_rpn_cls': loss_rpn_cls,
            'loss_rpn_box': loss_rpn_box,
            'loss_rcnn_cls': loss_rcnn_cls,
            'loss_rcnn_box': loss_rcnn_box,
            'loss_cls_distil_loss':loss_rcnn_cls_distil_loss,
            'loss_bbox_distil_loss':loss_rcnn_bbox_distil_loss,
            'loss_base_feat_distil_loss': loss_base_feat_distil_loss,
            #'loss_margin_loss':  loss_margin_loss

          }
          logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)

        loss_temp = 0
        start = time.time()

    if epoch==args.max_epochs or "coco" in args.dataset or epoch%10==0:
      save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
      save_checkpoint({
        'session': args.session,
        'epoch': epoch,
        'model': fasterRCNN_inc.module.state_dict() if args.mGPUs else fasterRCNN_inc.state_dict(),
        'optimizer': optimizer.state_dict(),
        'pooling_mode': cfg.POOLING_MODE,
        'class_agnostic': args.class_agnostic,
      }, save_name)
      print('save model: {}'.format(save_name))
      # save_name_res = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}_res.pth'.format(args.session, epoch, step))
      # save_checkpoint({
      #   'session': args.session,
      #   'epoch': epoch,
      #   'model': fasterRCNN_residual.module.state_dict() if args.mGPUs else fasterRCNN_residual.state_dict(),
      #   'optimizer': optimizer.state_dict(),
      #   'pooling_mode': cfg.POOLING_MODE,
      #   'class_agnostic': args.class_agnostic,
      # }, save_name_res)
      # print('save model: {}'.format(save_name_res))

  if args.use_tfboard:
    logger.close()
