# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from PIL import Image, ImageDraw
import numpy as np
import numpy.random as npr
# from scipy.misc import imread
from imageio import imread
import cv2
cv2read = True#False
from model.utils.config import cfg
from model.utils.blob import prep_im_for_blob, im_list_to_blob
import pdb
from .CopyPasteAugmenter import sample_1
import random
import torch
from torch.autograd import Variable
from roi_data_layer.roidb import combined_roidb
from model.utils.config import cfg, cfg_from_file, cfg_from_list

from model.faster_rcnn.resnet import resnet
from model.utils.augs import seq, superpixel, getFile
import copy

from model.utils.augs import mixup_data, mixup_criterion, getFile, mixup_data_img, mixup_data_old, mixup_criterion_kl, \
    croplist_mix, mixup_data_bg

np.random.seed(cfg.RNG_SEED)
torch.manual_seed(cfg.RNG_SEED)
torch.cuda.manual_seed(cfg.RNG_SEED)
torch.cuda.manual_seed_all(cfg.RNG_SEED)
random.seed(cfg.RNG_SEED)  ##
os.environ['PYTHONHASHSEED'] = str(cfg.RNG_SEED)
torch.backends.cudnn.deterministic = True

if cfg.TRAIN.BG=='img':
    croplist_bg = []
    if isinstance(cfg.TRAIN.CROP_PATH, list):
        crop_path_train = copy.deepcopy(cfg.TRAIN.CROP_PATH)
        crop_path_train.append(cfg.TRAIN.NEW_PATH)
    else:
        crop_path_train = [cfg.TRAIN.CROP_PATH, cfg.TRAIN.NEW_PATH]
    print('crops list: {}'.format(crop_path_train))
    print('cfg.TRAIN.CROP_PATH', cfg.TRAIN.CROP_PATH, 'crop_path_train', crop_path_train)
    if isinstance(crop_path_train, list):
        for crp in crop_path_train:
            getFile(crp, croplist_bg)
    else:
        getFile(crop_path_train, croplist_bg)


def get_minibatch(roidb, num_classes, crop_path=None, index=None):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert (cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
            format(num_images, cfg.TRAIN.BATCH_SIZE)

    if crop_path is not None:
      # Get the input image blob, formatted for caffe
      im_blob, im_scales = _get_image_blob_inc(roidb, random_scale_inds, crop_path=crop_path, index=index)
    else:
      # Get the input image blob, formatted for caffe
      im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

    blobs = {'data': im_blob}

    assert len(im_scales) == 1, "Single batch only"
    assert len(roidb) == 1, "Single batch only"

    # gt boxes: (x1, y1, x2, y2, cls)
    if cfg.TRAIN.USE_ALL_GT:
        # Include all ground truth boxes
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
    else:
        # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
        gt_inds = np.where((roidb[0]['gt_classes'] != 0) & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]

    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)

    try:
        gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
        gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
        # gt_boxes[roidb[0]['boxes'][gt_inds, :].shape[0], 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
        # gt_boxes[roidb[0]['boxes'][gt_inds, :].shape[0], 4] = roidb[0]['gt_classes'][gt_inds]
    except Exception as e:
        print('ydb except', e, gt_boxes.shape, roidb[0]['boxes'], roidb[0]['gt_classes'], gt_inds)
    if crop_path is not None and (cfg.TRAIN.CROP_CPGT or cfg.TRAIN.GRAY_CPGT):
        if cfg.TRAIN.CROP_CPGT:  ### if use crop gt, then gt is used for both crop and gray
          # print('ydb use gt for all crop image!!!!!')
          # print(roidb[0]['add_boxes'])
          if roidb[0]['add_boxes'] is not None:
              # try:
              add_boxes = np.array(roidb[0]['add_boxes'])
              if add_boxes.shape[0]>0:
                  add_boxes[:,0:4]= add_boxes[:,0:4]+[5,5,-5,-5]
                  add_boxes[:, 0:4] = add_boxes[:, 0:4] * im_scales[0]
                  gt_boxes = np.concatenate((gt_boxes, add_boxes), axis=0)
                  try:
                      roidb[0]['gt_classes'] = np.concatenate((roidb[0]['gt_classes'], add_boxes[:, 4]), axis=0)
                  except Exception as e:
                      print('except add_boxes', add_boxes, e)
                  if add_boxes.shape[0]>7:
                      print('ydb too many addboxes')
              else:
                  add_boxes = np.array(roidb[0]['add_boxes'])
                  print('add_bboxes<0', add_boxes)
        elif cfg.TRAIN.GRAY_CPGT and roidb[0]['image'] is None: ### if not use crop gt, but use GRAY gt, then gt is used for gray
          # print(roidb[0]['add_boxes'])
          # print('ydb use gt for gray image!!!!!')
          if roidb[0]['add_boxes'] is not None and len(np.array(roidb[0]['add_boxes']))>0:
              try:
                  add_boxes = np.array(roidb[0]['add_boxes'])
                  add_boxes[:, 0:4] = add_boxes[:, 0:4] * im_scales[0]
                  gt_boxes = np.concatenate((gt_boxes, add_boxes), axis=0)
                  roidb[0]['gt_classes'] = np.concatenate((roidb[0]['gt_classes'], add_boxes[:, 4]), axis=0)
              except:
                  add_boxes = np.array(roidb[0]['add_boxes'])
                  print('except add_boxes', add_boxes)


        # cfg.TRAIN.SAVEFLAG = True
        if cfg.TRAIN.SAVEFLAG:
            save_path = cfg.TRAIN.ROOT_P + 'CPR-IOD/faster-rcnn/save_cp/'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_name = save_path + str(random.randint(1, 1000)) + str(random.randint(1, 1000)) + '.jpg'
            draw = True  # False
            if draw:
                img = im_blob[0]+cfg.PIXEL_MEANS
                # print(img)
                img_c = img.copy()
                img_d = Image.fromarray(np.uint8(img_c[:, :, ::-1]))
                a = ImageDraw.ImageDraw(img_d)  # 用a来表示
                # 在边界框的两点（左上角、右下角）画矩形，无填充，边框红色，边框像素为5
                for box in gt_boxes:
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                    a.rectangle(((x1, y1), (x2, y2)), fill=None, outline='red')
                    a.text((x1, y1), str(box[4]), fill=(255, 255, 0))
                img_d.save(save_name)

                import matplotlib.pyplot as plt
                plt.figure("Image")
                plt.imshow(img_d)
                plt.axis('on')
                plt.title('image')
                plt.show()
            else:
                img = Image.fromarray(np.uint8(im_blob[0]+cfg.PIXEL_MEANS))
                img.save(save_name)
            img = np.array(img)
            print('ydb save', save_name)

    # if gt_boxes.shape[0]>10:
    #     print('ydb')
    if 'bg_coords' in roidb[0]:
        # try:
        bg_coords = np.array(roidb[0]['bg_coords'])
        bg_coords[:, 0:4] = bg_coords[:, 0:4] * im_scales[0]
        # # gt_boxes = np.concatenate((gt_boxes, bg_coords), axis=0)
        # roidb[0]['gt_classes'] = np.concatenate((roidb[0]['gt_classes'], bg_coords[:, 4]), axis=0)
        # except:
        #     add_boxes = np.array(roidb[0]['add_boxes'])
        #     print('except add_boxes', add_boxes)

    blobs['gt_boxes'] = gt_boxes
    blobs['im_info'] = np.array(
        [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
        dtype=np.float32)

    blobs['img_id'] = roidb[0]['img_id']
    if 'bg_coords' in roidb[0]:
        blobs['bg_coords'] = bg_coords
    return blobs

def _get_image_blob(roidb, scale_inds):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb)

  processed_ims = []
  im_scales = []
  for i in range(num_images):
    if not cv2read:
      im = imread(roidb[i]['image'])
    else:
      im = cv2.imread(roidb[i]['image'])[:, :, ::-1]
    # im = imread(roidb[i]['image'])

    if len(im.shape) == 2:
      im = im[:,:,np.newaxis]
      im = np.concatenate((im,im,im), axis=2)
    # flip the channel, since the original one using cv2
    # rgb -> bgr
    im = im[:,:,::-1]

    if roidb[i]['flipped']:
      im = im[:, ::-1, :]
    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                    cfg.TRAIN.MAX_SIZE)
    im_scales.append(im_scale)
    processed_ims.append(im)


  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, im_scales


def _get_image_blob_inc(roidb, scale_inds, crop_path=None, index=None):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    if roidb[0]['image']=='./data/VOCdevkit/VOC2007/JPEGImages/009699.jpg':
        print('wrongimage')
    cam=False#True
    if cam:
        features_blobs = []
        def hook_feature(module, input, output):
            features_blobs.append(output.data.cpu().numpy())
        blobs, im_scales = _get_image_blob(roidb, scale_inds)
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        im_data = torch.FloatTensor(1)
        im_info = torch.FloatTensor(1)
        num_boxes = torch.LongTensor(1)
        gt_boxes = torch.FloatTensor(1)
        cuda = 1
        # ship to cuda
        if cuda > 0:
            im_data = im_data.cuda()
            im_info = im_info.cuda()
            num_boxes = num_boxes.cuda()
            gt_boxes = gt_boxes.cuda()

        # make variable
        im_data = Variable(im_data, volatile=True)
        im_info = Variable(im_info, volatile=True)
        num_boxes = Variable(num_boxes, volatile=True)
        gt_boxes = Variable(gt_boxes, volatile=True)

        with torch.no_grad():
            im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
            im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
            gt_boxes.resize_(1, 1, 5).zero_()
            num_boxes.resize_(1).zero_()
        imdb_name_org = 'voc_2007_1_train_cat'  # 'voc_2007_trainval'#
        basefrcnn_load_name_org = '/mnt/disk7/ydb/Object_train/GFR-IL/faster-rcnn/model_baseline/cat/res50/cat/faster_rcnn_1_20_228.pth'
        basefrcnn_load_name_org = 'model_baseline/cat/res50/cat/faster_rcnn_1_20_228.pth'

        imdb_name_org = 'voc_2007_1_train_cattv'
        basefrcnn_load_name_org = 'model_baseline/cattv/res50/cattv/faster_rcnn_1_20_406.pth'
        # imdb_name_org = 'voc_2007_1_train_catdog'
        # basefrcnn_load_name_org = 'model_baseline/catdog/res50/catdog/faster_rcnn_1_20_508.pth'

        # basefrcnn_load_name_org = '/mnt/disk7/ydb/Object_train/GFR-IL/faster-rcnn/model_baseline/19_rpnse/res50/pascal_voc/faster_rcnn_1_20_3290.pth'#'model_baseline/models_res50_caffe_voc07_19/res50/pascal_voc/faster_rcnn_1_20_9873.pth'
        # basefrcnn_load_name_org = '/mnt/disk7/ydb/Object_train/GFR-IL/faster-rcnn/model_baseline/15_rpnse/res50/pascal_voc_07_15/faster_rcnn_1_20_3000.pth'
        imdb_org, roidb_org, ratio_list_org, ratio_index_org = combined_roidb(imdb_name_org)
        fasterRCNN_org = resnet(imdb_org.classes, 50, pretrained=True, class_agnostic=False)
        fasterRCNN_org.create_architecture()
        if cuda:
            fasterRCNN_org.cuda()
        print("load checkpoint %s" % (basefrcnn_load_name_org))
        if cuda > 0:
            checkpoint_org = torch.load(basefrcnn_load_name_org)
        else:
            checkpoint_org = torch.load(basefrcnn_load_name_org, map_location=(lambda storage, loc: storage))
        fasterRCNN_org.load_state_dict(checkpoint_org['model'])
        fasterRCNN_org.eval()
        # fasterRCNN_org._modules.get('RCNN_base').register_forward_hook(hook_feature)#
        fasterRCNN_org._modules.get('RCNN_rpn').RPN_Conv.register_forward_hook(hook_feature)
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, _ = fasterRCNN_org(im_data, im_info, gt_boxes, num_boxes)
        base_feat=features_blobs[0]
        rpn_conv1=torch.sum(torch.abs(base_feat),dim=2)
        att = torch.mean(torch.abs(rpn_conv1), dim=1)  # ,_ max
        att = att.view(rpn_conv1.shape[0], -1)
        min_v, _ = torch.min(att, dim=1)
        max_v, _ = torch.max(att, dim=1)
        att_norm = ((att - min_v.unsqueeze(dim=1).repeat(1, att.shape[1])) / (max_v - min_v + 0.000001).unsqueeze(dim=1).repeat(1, att.shape[1]))
        high_point = 0.8  # 0.8#0.8
        low_point = 0.1  # 0.1#0.1
        high_idx = (att_norm > high_point).nonzero()  # .view(-1)
        low_idx = (att_norm < low_point).nonzero()  # (
        low_new = rpn_conv1.view(rpn_conv1.shape[0], rpn_conv1.shape[1], -1)[:, :, low_idx[:, 1].long()]

    num_images = len(roidb)

    processed_ims = []
    im_scales = []
    for i in range(num_images):
        # im = cv2.imread(roidb[i]['image'])
        gray_aug = cfg.TRAIN.GRAY_AUG  # False#True
        # gray_roi = {'boxes': None, 'gt_classes': None, 'gt_ishard': None, 'gt_overlaps': None, 'flipped': False,
        #             'seg_areas': None, 'img_id': 558, \
        #             'image': None, 'width': 500, 'height': 500, 'max_classes': None, 'max_overlaps': None, 'need_crop': 0}
        # roidb[i] = gray_roi
        if gray_aug and roidb[i]['image'] is None:
            if index is not None and i%50==0:
                print('gray', index)
            # import random
            im = np.ones((roidb[i]['width'], roidb[i]['height'], 3), dtype=np.uint8) #(500, 500, 3)
            if cfg.TRAIN.BG == 'gray':
                im = im * random.randint(0, 255)
            elif cfg.TRAIN.BG == 'rand':
                im = im * np.random.randint(0,255,(im.shape), dtype=np.uint8) #
            elif cfg.TRAIN.BG == 'patch':
                patch_idx_w=random.randint(3, 6)
                # patch_idx_h = random.randint(1, 6)
                # patch_num = random.randint(1, 6)
                pt_size_w = int(im.shape[0]/patch_idx_w)
                pt_size_h = int(im.shape[1]/patch_idx_w)

                for idx in range(0, patch_idx_w):
                    x = random.randint(0, im.shape[0] - pt_size_w)
                    y = random.randint(0, im.shape[1] - pt_size_h)
                    pt = np.random.randint(0,255,(pt_size_w,pt_size_h,3), dtype=np.uint8)
                    im[x:x+pt_size_w,y:y+pt_size_h] = pt
            elif cfg.TRAIN.BG == 'img':
                im = superpixel(image=cv2.resize(cv2.imread(random.choice(croplist_bg))[:, :, ::-1],(roidb[i]['width'], roidb[i]['height'])))
                # bg_coords = mixup_data_bg(im, croplist_bg)
                # roidb[i]['bg_coords'] = bg_coords
                # ############################# draw gt box ##############################################
                # from PIL import Image, ImageDraw
                # img_c = im.copy()
                # img_d = Image.fromarray(np.uint8(img_c[:, :, ::-1]))
                # a = ImageDraw.ImageDraw(img_d)
                # import matplotlib.pyplot as plt
                # plt.figure("Image")
                # plt.imshow(img_d)
                # plt.axis('on')
                # plt.title('image')
                # plt.show()
                #####################################################################
            else:
                print('ydb unknown BG type!!!')
            # for widx in range(0, patch_idx_w):
            #     for hidx in range(0, patch_idx_h):
            #         for cidx in range(0,3):
            #             if widx==patch_idx_w-1 and hidx<patch_idx_w-1:
            #                 im[widx * pt_size_w:, hidx * pt_size_h:(hidx + 1) * pt_size_h,
            #                 cidx] *= random.randint(0, 255)
            #             elif  widx<patch_idx_w-1 and hidx==patch_idx_w-1:
            #                 im[widx * pt_size_w:(widx + 1) * pt_size_w, hidx * pt_size_h:,
            #                 cidx] *= random.randint(0, 255)
            #             elif widx==patch_idx_w-1 and hidx==patch_idx_w-1:
            #                 im[widx * pt_size_w:, hidx * pt_size_h:,
            #                 cidx] *= random.randint(0, 255)
            #             else:
            #                 im[widx*pt_size_w:(widx+1)*pt_size_w, hidx*pt_size_h:(hidx+1)*pt_size_h, cidx] *= random.randint(0, 255)


            gt_boxes = np.array([])
            gt_boxes = np.empty((0, 5), dtype=np.float32)
            if cfg.TRAIN.mixobj:
                im, cord_list_mix = sample_1.copy_paste_aug(im, crop_path=cfg.TRAIN.MIX_PATH, gt_boxes=gt_boxes,
                                                    save=False, gray_base_gt=cfg.TRAIN.mixnum, mix=True)  # #### gray 1 old,n new  cfg.TRAIN.GRAY_PATH
                if cord_list_mix is not None and len(cord_list_mix)>0:
                    if cord_list_mix[0][-1] == -2:
                        roidb[i]['bg_coords'] = np.array(cord_list_mix)

            im, cord_list = sample_1.copy_paste_aug(im, crop_path=cfg.TRAIN.BASE_PATH, gt_boxes=gt_boxes if not cfg.TRAIN.mixobj or 'bg_coords' not in roidb[i] else np.concatenate((gt_boxes, roidb[i]['bg_coords']),axis=0),
                                                        save=False, gray_base_gt=cfg.TRAIN.BASE_NUM)  # #### gray 1 old,n new  cfg.TRAIN.GRAY_PATH

            if cord_list is not None:
                try:
                    add_boxes = np.array(cord_list)
                    # add_boxes[:,0:4] = add_boxes[:, 0:4]
                    roidb[i]['boxes'] = add_boxes[:, :4]
                    roidb[i]['gt_classes'] = add_boxes[:, 4]
                except:
                    add_boxes = np.array(cord_list)
                    print('except add_boxes', add_boxes)
                    roidb[i]['boxes'] = None
                    roidb[i]['gt_classes'] = None

        if not (gray_aug and roidb[i]['image'] is None):
            # im = imread(roidb[i]['image'])
            if not cv2read:
                im = imread(roidb[i]['image'])
            else:
                im = cv2.imread(roidb[i]['image'])[:, :, ::-1]
            if len(im.shape) == 2:
                print('imshape==2',im.shape)
                im = im[:, :, np.newaxis]
                im = np.concatenate((im, im, im), axis=2)
        drop= 1#random.randint(0,1)

        if (drop and cfg.TRAIN.CROP_AUG and roidb[i]['image'] is not None) or (gray_aug and roidb[i]['image'] is None):#cfg.TRAIN.CROP_AUG or gray_aug:
            if cfg.TRAIN.USE_ALL_GT:
                # Include all ground truth boxes
                gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
            else:
                # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
                gt_inds = \
                np.where((roidb[0]['gt_classes'] != 0) & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
            # if len(gt_inds)>len(roidb[0]['boxes']):
            #     print('wrongima')
            gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
            # if roidb[0]['boxes'] is not None:###ydb
            try:
                gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :]
                gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
                # gt_boxes[roidb[0]['boxes'][gt_inds, :].shape[0], 0:4] = roidb[0]['boxes'][gt_inds, :]
                # gt_boxes[roidb[0]['boxes'][gt_inds, :].shape[0], 4] = roidb[0]['gt_classes'][gt_inds]
            except Exception as e:
                print('except gtboxes', e, gt_boxes)
            if roidb[i]['flipped']:
                im = im[:, ::-1, :]
            if gray_aug and roidb[i]['image'] is None: #### gray aug
                im, cord_list = sample_1.copy_paste_aug(im, crop_path=cfg.TRAIN.GRAY_PATH, gt_boxes=gt_boxes if not cfg.TRAIN.mixobj or 'bg_coords' not in roidb[i] else np.concatenate((gt_boxes, roidb[i]['bg_coords']),axis=0), save=cfg.TRAIN.SAVEFLAG,
                                                        gray=True)  # True
            elif cfg.TRAIN.CROP_AUG and roidb[i]['image'] is not None: ### new samples aug
                if cfg.TRAIN.mixobj and cfg.TRAIN.mixobj_crop:
                    im, cord_list_mix = sample_1.copy_paste_aug(im, crop_path=cfg.TRAIN.CROP_PATH, gt_boxes=gt_boxes,
                                                                save=False, gray_base_gt=cfg.TRAIN.mixnum, mix=True)  # #### gray 1 old,n new  cfg.TRAIN.GRAY_PATH
                    if cord_list_mix is not None and len(cord_list_mix)>0:
                        if cord_list_mix[0][-1] == -2:
                            roidb[i]['bg_coords'] = np.array(cord_list_mix)
                im, cord_list = sample_1.copy_paste_aug(im, crop_path=cfg.TRAIN.CROP_PATH, gt_boxes=gt_boxes if not cfg.TRAIN.mixobj or 'bg_coords' not in roidb[i] else np.concatenate((gt_boxes, roidb[i]['bg_coords']),axis=0), save=cfg.TRAIN.SAVEFLAG)#False)  # True

            if roidb[i]['flipped']:
                im = im[:, ::-1, :]
            roidb[0]['add_boxes'] = cord_list

        if len(im.shape) == 2:
            im = im[:, :, np.newaxis]
            im = np.concatenate((im, im, im), axis=2)
        # flip the channel, since the original one using cv2
        # rgb -> bgr
        im = im[:, :, ::-1]

        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales
