import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable

from model.utils.config import cfg
from model.rpn.rpn import _RPN
#from model.rpn.rpn_dl import _RPN
from model.rpn.rpn_distil_cts import _RPN_distil as _RPN ####################### rpn distil
# from model.rpn.rpn_distil import _RPN_distil_residual as _RPN ####################### rpn residual distil se block
from model.rpn.bbox_transform import bbox_transform_inv
from model.rpn.bbox_transform import clip_boxes
from model.roi_layers import ROIAlign, ROIPool
from model.roi_layers import nms
from model.utils.config import cfg
from model.utils.augs import mixup_data, mixup_criterion, getFile, mixup_data_img, mixup_data_old, mixup_criterion_kl, croplist_mix
# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_align.modules.roi_align import RoIAlignAvg
# from model.faster_rcnn.ISDA import EstimatorCV, ISDALoss
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
#from model.rpn.proposal_target_layer_cascade_distil import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

from PIL import Image, ImageDraw
import numpy as np

l2_loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
l1_loss_fn = torch.nn.L1Loss(reduce=True, size_average=True)

import torch.distributions as tdist
import torch
import copy
import os
np.random.seed(cfg.RNG_SEED)
torch.manual_seed(cfg.RNG_SEED)
torch.cuda.manual_seed(cfg.RNG_SEED)
torch.cuda.manual_seed_all(cfg.RNG_SEED)
random.seed(cfg.RNG_SEED)  ##
os.environ['PYTHONHASHSEED'] = str(cfg.RNG_SEED)
torch.backends.cudnn.deterministic = True

plot = False #True#
if cfg.TRAIN.CTS:
    from SupContrast.losses import SupConLoss
    temp = cfg.TRAIN.TMP  # 2#0.07
    base_tmp = temp
    criterion = SupConLoss(temperature=temp, contrast_mode='one', base_temperature=base_tmp)  # all,one

# if cfg.TRAIN.mixup:
#     croplist_mix = []
#     if isinstance(cfg.TRAIN.CROP_PATH, list):
#         crop_path_train = copy.deepcopy(cfg.TRAIN.CROP_PATH)
#         # crop_path_train.append(cfg.TRAIN.NEW_PATH)
#     else:
#         crop_path_train = [cfg.TRAIN.CROP_PATH]#[cfg.TRAIN.CROP_PATH, cfg.TRAIN.NEW_PATH]
#     print('crops list: {}'.format(crop_path_train))
#     print('cfg.TRAIN.CROP_PATH', cfg.TRAIN.CROP_PATH, 'crop_path_train', crop_path_train)
#     if isinstance(crop_path_train, list):
#         for crp in crop_path_train:
#             getFile(crp, croplist_mix)
#     else:
#         getFile(crop_path_train, croplist_mix)

def aug_gaussian_fea(loc = torch.Tensor([4.0,1.0,3.0,5.0,6.0]), scale = torch.Tensor([0.5,0.2,0.3,0.1,0.4]), fea_num=2):
    # loc = torch.Tensor([4.0, 1.0, 3.0, 5.0, 6.0])
    # scale = torch.Tensor([0.5, 0.2, 0.3, 0.1, 0.4])
    gaussian = tdist.Normal(loc,scale)
    aug_fea = gaussian.sample((fea_num,))
    return aug_fea

def correlation_distillation_loss(fea1, fea2, fea1_old, fea2_old):
    fea1 = fea1.reshape(fea1.shape[0], fea1.shape[1], -1)
    fea2 = fea2.reshape(fea2.shape[0], fea2.shape[1], -1)
    fea1_old = fea1_old.reshape(fea1_old.shape[0], fea1_old.shape[1], -1)
    fea2_old = fea2_old.reshape(fea2_old.shape[0], fea2_old.shape[1], -1)
    corr_loss = torch.Tensor([0]).cuda()
    for i in range(0, fea1.shape[0]):
        fea1_norm = F.normalize(fea1[i], dim=1)### dim=0 20210222ydb
        fea2_norm = F.normalize(fea2[i], dim=1)### dim=0 20210222ydb
        fea1_old_norm = F.normalize(fea1_old[i], dim=1)### dim=0 20210222ydb
        fea2_old_norm = F.normalize(fea2_old[i], dim=1)### dim=0 20210222ydb
        sim_matrix_org = fea2_old_norm.mm(fea1_old_norm.t())
        sim_matrix = fea2_norm.mm(fea1_norm.t())
        corr_loss += l1_loss_fn(sim_matrix, sim_matrix_org)
    return corr_loss

def correlation_distillation_loss_twodim(fea, fea_old):
    corr_loss = torch.Tensor([0]).cuda()
    fea_norm = F.normalize(fea, dim=1)### dim=0 20210222ydb
    fea_old_norm = F.normalize(fea_old, dim=1)### dim=0 20210222ydb
    sim_matrix_org = fea_old_norm.mm(fea_old_norm.t())
    sim_matrix = fea_norm.mm(fea_norm.t())
    corr_loss += l1_loss_fn(sim_matrix, sim_matrix_org)
    return corr_loss

def correlation_distillation_loss_twodim_2(fea, fea_2, fea_old, fea_old_2):
    corr_loss = torch.Tensor([0]).cuda()
    fea_norm = F.normalize(fea, dim=1)### dim=0 20210222ydb
    fea_old_norm = F.normalize(fea_old, dim=1)### dim=0 20210222ydb
    fea_norm_2 = F.normalize(fea_2, dim=1)  ### dim=0 20210222ydb
    fea_old_norm_2 = F.normalize(fea_old_2, dim=1)  ### dim=0 20210222ydb
    sim_matrix_org = fea_old_norm.mm(fea_old_norm_2.t())
    sim_matrix = fea_norm.mm(fea_norm_2.t())
    corr_loss += l1_loss_fn(sim_matrix, sim_matrix_org)
    return corr_loss

def SoftCrossEntropy(inputs, target, reduction='sum'):
    log_likelihood = -F.log_softmax(inputs, dim=1)
    batch = inputs.shape[0]
    if reduction == 'average':
        loss = torch.sum(torch.mul(log_likelihood, target)) / batch
    else:
        loss = torch.sum(torch.mul(log_likelihood, target))
    return loss


class CosineSimilarity(nn.Module):
    """
    This similarity function simply computes the cosine similarity between each pair of vectors. It has
    no parameters.
    """
    def forward(self, tensor_1, tensor_2):
        normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
        normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
        return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)

class L2Similarity(nn.Module):
    """
    This similarity function simply computes the l2 similarity between each pair of vectors. It has
    no parameters.
    """
    def forward(self, tensor_1, tensor_2):
        l2_loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
        l2_loss=0
        for t_1 in tensor_1:
            for t_2 in tensor_2:
                l2_loss+=l2_loss_fn(t_1,t_2)
        return l2_loss


def compute_iou(box1, box2, iou_thresh=0.3, wh=False):
    """
    compute the iou of two boxes.
    Args:
        box1, box2: [xmin, ymin, xmax, ymax] (wh=False) or [xcenter, ycenter, w, h] (wh=True)
        wh: the format of coordinate.
    Return:
        iou: iou of box1 and box2.
    """
    box1 = box1[0:4].cpu()
    box2 = box2[0:4].cpu()
    if wh == False:
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
    else:
        xmin1, ymin1 = int(box1[0] - box1[2] / 2.0), int(box1[1] - box1[3] / 2.0)
        xmax1, ymax1 = int(box1[0] + box1[2] / 2.0), int(box1[1] + box1[3] / 2.0)
        xmin2, ymin2 = int(box2[0] - box2[2] / 2.0), int(box2[1] - box2[3] / 2.0)
        xmax2, ymax2 = int(box2[0] + box2[2] / 2.0), int(box2[1] + box2[3] / 2.0)

    ## 获取矩形框交集对应的左上角和右下角的坐标（intersection）
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    ## 计算两个矩形框面积
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1])) # 计算交集面积
    iou = inter_area / (area1 + area2 - inter_area + 1e-6) #计算交并比
    if iou>iou_thresh:
        return True
    else:
        return False
    #return iou

def compute_iou_iou(box1, box2, iou_thresh=0.3, wh=False):
    """
    compute the iou of two boxes.
    Args:
        box1, box2: [xmin, ymin, xmax, ymax] (wh=False) or [xcenter, ycenter, w, h] (wh=True)
        wh: the format of coordinate.
    Return:
        iou: iou of box1 and box2.
    """
    box1 = box1[0:4]
    box2 = box2[0:4]

    if wh == False:
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
    else:
        xmin1, ymin1 = int(box1[0] - box1[2] / 2.0), int(box1[1] - box1[3] / 2.0)
        xmax1, ymax1 = int(box1[0] + box1[2] / 2.0), int(box1[1] + box1[3] / 2.0)
        xmin2, ymin2 = int(box2[0] - box2[2] / 2.0), int(box2[1] - box2[3] / 2.0)
        xmax2, ymax2 = int(box2[0] + box2[2] / 2.0), int(box2[1] + box2[3] / 2.0)

    ## 获取矩形框交集对应的左上角和右下角的坐标（intersection）
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    ## 计算两个矩形框面积
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1])) # 计算交集面积
    iou = inter_area / (area1 + area2 - inter_area + 1e-6) #计算交并比
    iot = min(inter_area/area1, inter_area/area2)
    return iou, iot

class _fasterRCNN_inc_bbox_distil(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN_inc_bbox_distil, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.n_new_class =  cfg.NEW_CLASSES#10#5#1#5#5#10 #5 ############################# inc class num
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)

    def forward(self, im_data, im_info, gt_boxes, num_boxes, rois_org=None, cls_prob_org=None, bbox_pred_org=None, rois_label_org=None, fasterRCNN_org=None, step=None, roidb=None, ratio_index=None, prototypes=None, stds=None, fasterRCNN_residual=None, channel_att_old=None, se_flag=False, crd_criterion=None, opt=None, prototype_label=None,
                isda_criterion=None, epoch=None, isda_tmp=None, exsup=False, gray_flag=False, bg_coords=None, rois_inc = None, cls_prob_inc=None, bbox_pred_inc=None):
        imglevel = True#False#
        if cfg.TRAIN.mixup and imglevel and len(gt_boxes[0][:num_boxes].nonzero().view(-1))>0:
            class_mix = [self.classes.index(idx_crop.split('/')[-2]) for idx_crop in croplist_mix]
            img_mix = im_data[0].permute(1, 2, 0) + torch.from_numpy(cfg.PIXEL_MEANS).cuda().float()
            mix_im, mix_gt_boxes, mixup_lam = mixup_data_img(img_mix, gt_boxes[0][:1], croplist_mix, class_mix, alpha=1)#0.2
            mix_gt_boxes = mix_gt_boxes.unsqueeze(dim=0)
            mix_im = mix_im - torch.from_numpy(cfg.PIXEL_MEANS).cuda().float()
            mix_im = mix_im.permute(2,0,1).unsqueeze(dim=0)
            mix_base_feat = self.RCNN_base(mix_im)
            mix_base_feat_org = fasterRCNN_org.RCNN_base(mix_im)

            # mix_rois_a, mix_rpn_loss_cls_a, mix_rpn_loss_bbox_a, mix_att_loss_a = self.RCNN_rpn(mix_base_feat, im_info, mix_gt_boxes[:,:,:-1], num_boxes)  #### rpn original
            # mix_rois_b, mix_rpn_loss_cls_b, mix_rpn_loss_bbox_b, mix_att_loss_b = self.RCNN_rpn(mix_base_feat, im_info, torch.cat((mix_gt_boxes[:, :, :-2],mix_gt_boxes[:, :, -1]), dim=-1), num_boxes)  #### rpn original

            mix_gt_boxes_roi = mix_gt_boxes.new(mix_gt_boxes.size()).zero_()
            mix_gt_boxes_roi[:, :, 1:5] = mix_gt_boxes[:, :, :4]
            mix_rois_gt = mix_gt_boxes_roi[:, :num_boxes,:5]
            mix_rois_label_gt_a = Variable(mix_gt_boxes[:, :num_boxes, -2].view(-1).long())
            mix_rois_label_gt_b = Variable(mix_gt_boxes[:, :num_boxes, -1].view(-1).long())

            if plot:
                ############################# draw gt box ##############################################
                from PIL import Image, ImageDraw
                img = (mix_im[0].permute(1, 2, 0) + torch.from_numpy(cfg.PIXEL_MEANS).cuda().float()).cpu().numpy()
                print(img)
                img_c = img.copy()
                img_d = Image.fromarray(np.uint8(img_c[:, :, ::-1]))
                a = ImageDraw.ImageDraw(img_d)
                for box in mix_gt_boxes.cpu().numpy()[0]:
                    x1, y1, x2, y2 = box[0] + 10, box[1] + 10, box[2] - 10, box[3] - 10
                    a.rectangle(((x1, y1), (x2, y2)), fill=None, outline='red')
                    a.text((x1, y1), str(box[4]), fill=(255, 255, 0))
                import matplotlib.pyplot as plt
                plt.figure("Image")
                plt.imshow(img_d)
                plt.axis('on')
                plt.title('image')
                plt.show()
                #####################################################################
        ee2 = time.time()
        batch_size = im_data.size(0)
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # gt_add_old = torch.zeros(1,20,5)
        # gt_new = torch.zeros(1,20,5)
        # for bs in range(gt_boxes.shape[0]):
        #     gt_add_old = gt_boxes[bs][((gt_boxes[bs][:, 4] < (self.n_classes - self.n_new_class)) & (gt_boxes[bs][:, 4] >0)).nonzero().view(-1)]
        #     gt_new = gt_boxes[bs][(gt_boxes[bs][:, 4] >= (self.n_classes - self.n_new_class)).nonzero().view(-1)]

        l2_loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
        l1_loss_fn = torch.nn.L1Loss(reduce=True, size_average=True)

        ############################## draw gt box ##############################################

        #from PIL import Image, ImageDraw
        ## import cv2
        ## im=cv2.imread(roidb[ratio_index[step]]['image'])
        #
        ## im = cv2.resize(im, (int(im_info[0][1].cpu().item()),int(im_info[0][0].cpu().item())),
        ##                interpolation=cv2.INTER_LINEAR)
        #
        ## 创建一个可以在给定图像上绘图的对象
        #img = Image.open(roidb[ratio_index[step]]['image'])
        #img = img.resize((int(im_info[0][1].cpu().item()), int(im_info[0][0].cpu().item())))
        #import numpy as np
        #if roidb[ratio_index[step]]['flipped']:
        #    img = Image.fromarray(np.array(img)[:, ::-1, :])
        #draw = ImageDraw.Draw(img)
        #
        #for idx, g in enumerate(gt_boxes.squeeze()):
        #    if idx == num_boxes.cpu().item():
        #        break
        #    box = g.cpu()
        #    # cv2.rectangle(im, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (255, 0, 0), -1)
        #    # cv2.putText(im, self.classes[int(box[4].item())],  (box[0], box[1]),  cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 255), 2)
        #    draw.rectangle((box[0], box[1], box[2], box[3]), outline=(255, 0, 0))
        #    draw.text([box[0], box[1]], self.classes[int(box[4].item())], "red")
        ## cv2.imwrite('1.jpg',im)
        ##img.save('2.jpg')
        #img.save('imgs_save_thesis/' + str(random.randint(0, 100000)) + '.jpg')

        ###################################################################################################################
        if cfg.TRAIN.IM_AUG or (cfg.TRAIN.PSE is False) or rois_org is None:# or (cfg.TRAIN.GRAY_AUG is False):#cfg.TRAIN.IM_AUG or
            use_pesudo = False
            # print('ydb not use pseudo')
        else:
            use_pesudo = True#False#True
        if use_pesudo:# and not gray_flag:
            ########## frcnn_org_result #################
            scores = cls_prob_org.data
            boxes = rois_org.data[:, :, 1:5]
            batch_size = im_data.shape[0]

            if cfg.TEST.BBOX_REG:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred_org.data
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                    # Optionally normalize targets by a precomputed mean and stdev
                    if self.class_agnostic:
                        if torch.cuda.is_available():
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                            box_deltas = box_deltas.view(batch_size, -1, 4)  # box_deltas = box_deltas.view(1, -1, 4)
                        else:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                                cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                            box_deltas = box_deltas.view(batch_size, -1, 4)  # box_deltas = box_deltas.view(1, -1, 4)
                    else:
                        if torch.cuda.is_available():
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                            box_deltas = box_deltas.view(batch_size, -1, 4 * (
                            fasterRCNN_org.n_classes))  # self.n_classes-1  box_deltas = box_deltas.view(1, -1, 4 * (fasterRCNN_org.n_classes))#self.n_classes-1
                        else:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                                cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                            box_deltas = box_deltas.view(batch_size, -1, 4 * (
                                fasterRCNN_org.n_classes))  # self.n_classes-1  box_deltas = box_deltas.view(1, -1, 4 * (fasterRCNN_org.n_classes))#self.n_classes-1

                pred_boxes = bbox_transform_inv(boxes, box_deltas, batch_size)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, batch_size)
            else:
                # Simply repeat the boxes, once for each class
                pred_boxes = np.tile(boxes, (1, scores.shape[1]))

            # pred_boxes /= im_info.data[0][2]
            thr_2 = False  # cfg.threshold_2
            if thr_2:
                thresh = 0.9
            else:
                thresh = cfg.TRAIN.pseth#0.9  # 9#9#9  # 0.5#0.1#0.7#0.1#0.5#0.3#

            for bs_idx in range(scores.shape[0]):
                score_bs = scores[bs_idx].squeeze(dim=0)
                pred_box_bs = pred_boxes[bs_idx].squeeze(dim=0)

                # print(scores.shape)
                if torch.cuda.is_available():
                    org_det_gt_boxes = torch.Tensor().cuda()
                else:
                    org_det_gt_boxes = torch.Tensor()
                # try:
                #     print(scores.shape[1])
                # except:
                #     print(1)
                for j in range(1, score_bs.shape[1]):
                    inds = torch.nonzero(score_bs[:, j] > thresh).view(-1)
                    # if there is det
                    if inds.numel() > 0:
                        cls_scores = score_bs[:, j][inds]
                        _, order = torch.sort(cls_scores, 0, True)
                        if self.class_agnostic:
                            cls_boxes = pred_box_bs[inds, :]
                        else:
                            cls_boxes = pred_box_bs[inds][:, j * 4:(j + 1) * 4]

                        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                        # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                        cls_dets = cls_dets[order]
                        keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)  # NMS=0.3
                        cls_dets = cls_dets[keep.view(-1).long()]
                        if torch.cuda.is_available():
                            cls_tensor = torch.full([cls_dets.shape[0], 1], j).cuda()
                        else:
                            cls_tensor = torch.full([cls_dets.shape[0], 1], j)
                        cls_label_cat = torch.cat((cls_dets[:, 0:4], cls_tensor), 1)
                        if org_det_gt_boxes.shape[0] == 0:
                            org_det_gt_boxes = cls_label_cat
                        else:
                            org_det_gt_boxes = torch.cat((org_det_gt_boxes, cls_label_cat), 0)

                #############################################


                ####compute IOU between gt_boxes and rpn_org_proposals, delete overlapped bboxes of rpn_org_proposals
                if torch.cuda.is_available():
                    final_org_det = torch.Tensor().cuda()
                    final_org_det_re = torch.Tensor().cuda()
                else:
                    final_org_det = torch.Tensor()
                    final_org_det_re = torch.Tensor()

                if cfg.TRAIN.CROP_CPGT is True and cfg.TRAIN.Filter is True:
                    gt_new = gt_boxes[bs_idx][(gt_boxes[bs_idx][:, 4] >= (self.n_classes - self.n_new_class)).nonzero().view(-1)]
                    for o_bbox in org_det_gt_boxes:
                        uq_flag = True
                        for gt_bbox in gt_new:
                            if compute_iou(o_bbox, gt_bbox):  # iou=0.3
                                uq_flag = False
                                break
                        if uq_flag:
                            if final_org_det.shape[0] == 0:
                                final_org_det = o_bbox.unsqueeze(dim=0)
                            else:
                                final_org_det = torch.cat((final_org_det, o_bbox.unsqueeze(dim=0)), 0)
                    refine=cfg.TRAIN.Filter
                    gtflag=True
                    if refine:
                        gt_add_old = gt_boxes[bs_idx][((gt_boxes[bs_idx][:, 4] < (self.n_classes - self.n_new_class)) & (gt_boxes[bs_idx][:, 4] > 0)).nonzero().view(-1)]
                        f_bbox_flag = torch.Tensor([False for fflag in range(0, final_org_det.shape[0])]).cuda()
                        final_org_det_cp = final_org_det.clone()
                        go_flag = torch.Tensor([False for fflag in range(0, gt_add_old.shape[0])]).cuda()
                        for ff_idx, f_bbox in enumerate(final_org_det):
                            max_iou=0
                            max_iot=0
                            gt_add_old_c = gt_add_old[(gt_add_old[:,4]==f_bbox[4]).nonzero().view(-1)]
                            for go_idx, go_bbox in enumerate(gt_add_old_c):
                                # go_bbox[0], go_bbox[1], go_bbox[2], go_bbox[3] = go_bbox[0]+10, go_bbox[1]+10, go_bbox[2]-10, go_bbox[3]-10
                                iou,iot = compute_iou_iou(go_bbox, f_bbox)
                                if iot>0.9 and iou>0.9 :  # iou=0.7,compute_iou(go_bbox, f_bbox, iou_thresh=0.5)
                                    # if go_flag[go_idx]==False:
                                    if iot>max_iot and iou>max_iou:
                                        final_org_det[ff_idx] = go_bbox
                                        if final_org_det_re.shape[0] == 0:
                                            final_org_det_re = go_bbox.unsqueeze(dim=0)
                                        else:
                                            final_org_det_re = torch.cat((final_org_det_re, go_bbox.unsqueeze(dim=0)), 0)
                                        f_bbox_flag[ff_idx]=True
                                        print('refine bbox for pse!!!',iou, iot, f_bbox[4], max_iou, max_iot, (final_org_det[ff_idx]+go_bbox)/2-final_org_det[ff_idx])
                                        # final_org_det[ff_idx] = (final_org_det[ff_idx]+go_bbox)/2
                                        max_iot=iot
                                        max_iou=iou
                                #     # go_flag[go_idx]=True
                                # elif f_bbox_flag[ff_idx]==False:
                                #     add_temp = f_bbox
                                #     f_bbox_flag[ff_idx]=True
                                # else:
                                #     continue
                                # if final_org_det_re.shape[0] == 0:
                                #     final_org_det_re = add_temp.unsqueeze(dim=0)
                                # else:
                                #     final_org_det_re = torch.cat((final_org_det_re, add_temp.unsqueeze(dim=0)), 0)
                        if not gtflag:
                            final_org_det_re = final_org_det#[(f_bbox_flag==True).nonzero().view(-1)]
                        # print('filter before', final_org_det.shape[0],'filter after',(f_bbox_flag==True).nonzero().view(-1).shape[0] )
                    else:
                        final_org_det_re = final_org_det

                    # print('filter before', 'gt_old', (gt_add_old.shape), 'gt_new', gt_new.shape, 'ps',
                    #       org_det_gt_boxes.shape)

                    org_det_gt_boxes = torch.cat((gt_new, final_org_det_re),dim=0)  # .unsqueeze(dim=0)

                    if gt_boxes.shape[1]-org_det_gt_boxes.shape[0]>0:
                        zeropad = torch.zeros(gt_boxes.shape[1]-org_det_gt_boxes.shape[0], 5).cuda()
                        gt_boxes[bs_idx] = torch.cat((org_det_gt_boxes, zeropad), dim=0)
                    else:
                        gt_boxes_cp = torch.zeros(gt_boxes.shape[0],org_det_gt_boxes.shape[0],5).cuda()
                        gt_boxes_cp[:gt_boxes.shape[0],:gt_boxes.shape[1],:gt_boxes.shape[2]] = gt_boxes
                        gt_boxes_cp[bs_idx] = org_det_gt_boxes
                        gt_boxes = gt_boxes_cp
                        # gt_boxes[bs_idx] = org_det_gt_boxes[:gt_boxes[bs_idx].shape[0]]
                    ###################################################
                    # print('filter after', (gt_boxes[:,:,4]!=0).nonzero().shape,org_det_gt_boxes.shape[0])
                    num_boxes[bs_idx] = org_det_gt_boxes.shape[0]  ### not use pesudo
                    ###########################################################
                else:
                    for o_bbox in org_det_gt_boxes:
                        uq_flag = True
                        for gt_bbox in gt_boxes[bs_idx].squeeze()[:num_boxes[bs_idx]]:
                            if compute_iou(o_bbox, gt_bbox):  # iou=0.3
                                # print('pseudo overlap with gt',o_bbox, gt_bbox)
                                uq_flag = False
                                break
                        if uq_flag:
                            if final_org_det.shape[0] == 0:
                                final_org_det = o_bbox.unsqueeze(dim=0)
                            else:
                                final_org_det = torch.cat((final_org_det, o_bbox.unsqueeze(dim=0)), 0)
                    org_det_gt_boxes = final_org_det  # .unsqueeze(dim=0)
                    ###################################################
                    gt_boxes_n = gt_boxes.clone()
                    num_boxes_n = num_boxes.clone()
                    gt_boxes_bs = torch.cat((org_det_gt_boxes, gt_boxes[bs_idx]), 0)#.unsqueeze(dim=0)  ### not use pesudo
                    num_boxes[bs_idx] += org_det_gt_boxes.shape[0]  ### not use pesudo
                    gt_boxes[bs_idx]=gt_boxes_bs[:gt_boxes_n.shape[1]]

                    ###########################################################

        use_pesudo_new = cfg.TRAIN.PSE_NEW#False#
        if use_pesudo_new and rois_inc is not None:# and not gray_flag:
            ########## frcnn_inc_eval_result #################
            scores_inc = cls_prob_inc.data
            boxes_inc = rois_inc.data[:, :, 1:5]
            if cfg.TEST.BBOX_REG:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred_inc.data
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                    # Optionally normalize targets by a precomputed mean and stdev
                    if self.class_agnostic:
                        if torch.cuda.is_available():
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                            box_deltas = box_deltas.view(batch_size, -1, 4)  # box_deltas = box_deltas.view(1, -1, 4)
                        else:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                                cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                            box_deltas = box_deltas.view(batch_size, -1, 4)  # box_deltas = box_deltas.view(1, -1, 4)
                    else:
                        if torch.cuda.is_available():
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                            box_deltas = box_deltas.view(batch_size, -1, 4 * (
                            self.n_classes))  # self.n_classes-1  box_deltas = box_deltas.view(1, -1, 4 * (fasterRCNN_org.n_classes))#self.n_classes-1
                        else:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                                cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                            box_deltas = box_deltas.view(batch_size, -1, 4 * (
                                self.n_classes))  # self.n_classes-1  box_deltas = box_deltas.view(1, -1, 4 * (fasterRCNN_org.n_classes))#self.n_classes-1

                pred_boxes = bbox_transform_inv(boxes_inc, box_deltas, batch_size)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, batch_size)
            else:
                # Simply repeat the boxes, once for each class
                pred_boxes = np.tile(boxes_inc, (1, scores_inc.shape[1]))

            # pred_boxes /= im_info.data[0][2]
            thr_2 = False  # cfg.threshold_2
            if thr_2:
                thresh = 0.9
            else:
                thresh = cfg.TRAIN.pseth#0.9  # 9#9#9  # 0.5#0.1#0.7#0.1#0.5#0.3#

            for bs_idx in range(scores_inc.shape[0]):
                score_bs = scores_inc[bs_idx].squeeze(dim=0)
                pred_box_bs = pred_boxes[bs_idx].squeeze(dim=0)

                # print(scores_inc.shape)
                if torch.cuda.is_available():
                    org_det_gt_boxes = torch.Tensor().cuda()
                else:
                    org_det_gt_boxes = torch.Tensor()
                # try:
                #     print(scores_inc.shape[1])
                # except:
                #     print(1)
                for j in range(self.n_classes-self.n_new_class, score_bs.shape[1]):
                    inds = torch.nonzero(score_bs[:, j] > thresh).view(-1)
                    # if there is det
                    if inds.numel() > 0:
                        cls_scores = score_bs[:, j][inds]
                        _, order = torch.sort(cls_scores, 0, True)
                        if self.class_agnostic:
                            cls_boxes = pred_box_bs[inds, :]
                        else:
                            cls_boxes = pred_box_bs[inds][:, j * 4:(j + 1) * 4]

                        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                        # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                        cls_dets = cls_dets[order]
                        keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)  # NMS=0.3
                        cls_dets = cls_dets[keep.view(-1).long()]
                        if torch.cuda.is_available():
                            cls_tensor = torch.full([cls_dets.shape[0], 1], j).cuda()
                        else:
                            cls_tensor = torch.full([cls_dets.shape[0], 1], j)
                        cls_label_cat = torch.cat((cls_dets[:, 0:4], cls_tensor), 1)
                        if org_det_gt_boxes.shape[0] == 0:
                            org_det_gt_boxes = cls_label_cat
                        else:
                            org_det_gt_boxes = torch.cat((org_det_gt_boxes, cls_label_cat), 0)

                #############################################


                ####compute IOU between gt_boxes and rpn_org_proposals, delete overlapped bboxes of rpn_org_proposals
                if torch.cuda.is_available():
                    final_org_det = torch.Tensor().cuda()
                    final_org_det_re = torch.Tensor().cuda()
                else:
                    final_org_det = torch.Tensor()
                    final_org_det_re = torch.Tensor()

                if cfg.TRAIN.CROP_CPGT is True and cfg.TRAIN.Filter is True:
                    gt_new = gt_boxes[bs_idx][(gt_boxes[bs_idx][:, 4] >= (self.n_classes - self.n_new_class)).nonzero().view(-1)]
                    for o_bbox in org_det_gt_boxes:
                        uq_flag = True
                        for gt_bbox in gt_new:
                            if compute_iou(o_bbox, gt_bbox):  # iou=0.3
                                uq_flag = False
                                break
                        if uq_flag:
                            if final_org_det.shape[0] == 0:
                                final_org_det = o_bbox.unsqueeze(dim=0)
                            else:
                                final_org_det = torch.cat((final_org_det, o_bbox.unsqueeze(dim=0)), 0)
                    refine=cfg.TRAIN.Filter
                    gtflag=True
                    if refine:
                        gt_add_old = gt_boxes[bs_idx][((gt_boxes[bs_idx][:, 4] < (self.n_classes - self.n_new_class)) & (gt_boxes[bs_idx][:, 4] > 0)).nonzero().view(-1)]
                        f_bbox_flag = torch.Tensor([False for fflag in range(0, final_org_det.shape[0])]).cuda()
                        final_org_det_cp = final_org_det.clone()
                        go_flag = torch.Tensor([False for fflag in range(0, gt_add_old.shape[0])]).cuda()
                        for ff_idx, f_bbox in enumerate(final_org_det):
                            max_iou=0
                            max_iot=0
                            gt_add_old_c = gt_add_old[(gt_add_old[:,4]==f_bbox[4]).nonzero().view(-1)]
                            for go_idx, go_bbox in enumerate(gt_add_old_c):
                                # go_bbox[0], go_bbox[1], go_bbox[2], go_bbox[3] = go_bbox[0]+10, go_bbox[1]+10, go_bbox[2]-10, go_bbox[3]-10
                                iou,iot = compute_iou_iou(go_bbox, f_bbox)
                                if iot>0.9 and iou>0.9 :  # iou=0.7,compute_iou(go_bbox, f_bbox, iou_thresh=0.5)
                                    # if go_flag[go_idx]==False:
                                    if iot>max_iot and iou>max_iou:
                                        final_org_det[ff_idx] = go_bbox
                                        if final_org_det_re.shape[0] == 0:
                                            final_org_det_re = go_bbox.unsqueeze(dim=0)
                                        else:
                                            final_org_det_re = torch.cat((final_org_det_re, go_bbox.unsqueeze(dim=0)), 0)
                                        f_bbox_flag[ff_idx]=True
                                        print('refine bbox for pse!!!',iou, iot, f_bbox[4], max_iou, max_iot, (final_org_det[ff_idx]+go_bbox)/2-final_org_det[ff_idx])
                                        # final_org_det[ff_idx] = (final_org_det[ff_idx]+go_bbox)/2
                                        max_iot=iot
                                        max_iou=iou
                                #     # go_flag[go_idx]=True
                                # elif f_bbox_flag[ff_idx]==False:
                                #     add_temp = f_bbox
                                #     f_bbox_flag[ff_idx]=True
                                # else:
                                #     continue
                                # if final_org_det_re.shape[0] == 0:
                                #     final_org_det_re = add_temp.unsqueeze(dim=0)
                                # else:
                                #     final_org_det_re = torch.cat((final_org_det_re, add_temp.unsqueeze(dim=0)), 0)
                        if not gtflag:
                            final_org_det_re = final_org_det#[(f_bbox_flag==True).nonzero().view(-1)]
                        # print('filter before', final_org_det.shape[0],'filter after',(f_bbox_flag==True).nonzero().view(-1).shape[0] )
                    else:
                        final_org_det_re = final_org_det

                    # print('filter before', 'gt_old', (gt_add_old.shape), 'gt_new', gt_new.shape, 'ps',
                    #       org_det_gt_boxes.shape)

                    org_det_gt_boxes = torch.cat((gt_new, final_org_det_re),dim=0)  # .unsqueeze(dim=0)

                    if gt_boxes.shape[1]-org_det_gt_boxes.shape[0]>0:
                        zeropad = torch.zeros(gt_boxes.shape[1]-org_det_gt_boxes.shape[0], 5).cuda()
                        gt_boxes[bs_idx] = torch.cat((org_det_gt_boxes, zeropad), dim=0)
                    else:
                        gt_boxes_cp = torch.zeros(gt_boxes.shape[0],org_det_gt_boxes.shape[0],5).cuda()
                        gt_boxes_cp[:gt_boxes.shape[0],:gt_boxes.shape[1],:gt_boxes.shape[2]] = gt_boxes
                        gt_boxes_cp[bs_idx] = org_det_gt_boxes
                        gt_boxes = gt_boxes_cp
                        # gt_boxes[bs_idx] = org_det_gt_boxes[:gt_boxes[bs_idx].shape[0]]
                    ###################################################
                    # print('filter after', (gt_boxes[:,:,4]!=0).nonzero().shape,org_det_gt_boxes.shape[0])
                    num_boxes[bs_idx] = org_det_gt_boxes.shape[0]  ### not use pesudo
                    ###########################################################
                else:
                    for o_bbox in org_det_gt_boxes:
                        uq_flag = True
                        for gt_bbox in gt_boxes[bs_idx].squeeze()[:num_boxes[bs_idx]]:
                            if compute_iou(o_bbox, gt_bbox):  # iou=0.3
                                # print('pseudo overlap with gt',o_bbox, gt_bbox)
                                uq_flag = False
                                break
                        if uq_flag:
                            if final_org_det.shape[0] == 0:
                                final_org_det = o_bbox.unsqueeze(dim=0)
                            else:
                                final_org_det = torch.cat((final_org_det, o_bbox.unsqueeze(dim=0)), 0)
                    org_det_gt_boxes = final_org_det  # .unsqueeze(dim=0)
                    ###################################################
                    gt_boxes_n = gt_boxes.clone()
                    num_boxes_n = num_boxes.clone()
                    gt_boxes_bs = torch.cat((org_det_gt_boxes, gt_boxes[bs_idx]), 0)#.unsqueeze(dim=0)  ### not use pesudo
                    num_boxes[bs_idx] += org_det_gt_boxes.shape[0]  ### not use pesudo
                    gt_boxes[bs_idx]=gt_boxes_bs[:gt_boxes_n.shape[1]]

                    ###########################################################
        e22 = time.time()

        # def tensor_to_PIL(tensor):
        #     image = tensor.cpu().clone()
        #     image = image.squeeze(0)
        #     from torchvision import transforms
        #     unloader = transforms.ToPILImage()
        #     image = unloader(image)
        #     return image
        # img=tensor_to_PIL(im_data)
        plot=True
        if plot:
            from PIL import Image, ImageDraw
            import numpy as np
            ############################# draw gt box ##############################################
            img = (im_data[0].permute(1,2,0) + torch.from_numpy(cfg.PIXEL_MEANS).cuda().float()).cpu().numpy()
            # print(img)
            img_c = img.copy()
            img_d = Image.fromarray(np.uint8(img_c[:, :, ::-1]))
            a = ImageDraw.ImageDraw(img_d)
            for box in gt_boxes.cpu().numpy()[0][:num_boxes[0]]:
                x1, y1, x2, y2 = box[0]+5, box[1]+5, box[2]-5, box[3]-5
                a.rectangle(((x1, y1), (x2, y2)), fill=None, outline='red')
                a.text((x1, y1), self.classes[int(box[4])], fill=(255, 255, 0))
            #if bg_coords is not None:
            #    for box in bg_coords.cpu().numpy()[0]:
            #        x1, y1, x2, y2 = box[0] + 10, box[1] + 10, box[2] - 10, box[3] - 10
            #        a.rectangle(((x1, y1), (x2, y2)), fill=None, outline='red')
            #        a.text((x1, y1), 'bg', fill=(255, 255, 0))
            import matplotlib.pyplot as plt
            plt.figure("Image")
            plt.imshow(img_d)
            plt.axis('on')
            plt.title('image')
            plt.show()
            img_d.save('imgs_save_thesis/' + str(random.randint(0, 100000)) + '.jpg')
            #######################################################################################
        #
        #
        #import cv2
        #im=cv2.imread(roidb[ratio_index[step]]['image'])

        #im = cv2.resize(im, (int(im_info[0][1].cpu().item()),int(im_info[0][0].cpu().item())),
        #                interpolation=cv2.INTER_LINEAR)

        # 创建一个可以在给定图像上绘图的对象
        #from PIL import Image, ImageDraw
        #import numpy as np
        #try:
        #    img=Image.open(roidb[ratio_index[step]]['image'])
        #    img=img.resize((int(im_info[0][1].cpu().item()),int(im_info[0][0].cpu().item())))
        #    import numpy as np
        #    if roidb[ratio_index[step]]['flipped']:
        #        img = Image.fromarray(np.array(img)[:, ::-1, :])
        #    draw = ImageDraw.Draw(img)

        #    for idx,g in enumerate(gt_boxes.squeeze()):
        #        if idx==num_boxes.cpu().item():
        #            break
        #        box=g.cpu()
        #        #cv2.rectangle(im, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (255, 0, 0), -1)
        #        #cv2.putText(im, self.classes[int(box[4].item())],  (box[0], box[1]),  cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 255), 2)
        #        draw.rectangle((box[0], box[1], box[2], box[3]), outline=(255, 0, 0))
        #        draw.text([box[0], box[1]], self.classes[int(box[4].item())], "red")
        #    #cv2.imwrite('1.jpg',im)
        #    # img.save('1.jpg')
        #    img.save('imgs_save_thesis/' + str(random.randint(0, 100000)) + '.jpg')
        #except Exception as e:
        #    print(e)

        #
        # ##################################################################################################################


        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        # if se_flag:
        # base_feat, c_att = self.channel_att(base_feat)
        # else:
        #     c_att = None

        #base_feat_res = fasterRCNN_residual.RCNN_base(im_data)
        ##################### feature distil ###############################
        base_feat_org = fasterRCNN_org.RCNN_base(im_data)
        # base_feat_org, c_att_org = fasterRCNN_org.channel_att(base_feat_org)

        # max_a = max(c_att_org.view(-1))
        # min_a = min(c_att_org.view(-1))
        # channel_att_norm = (((c_att_org.view(-1)) - min_a) / (max_a - min_a + 0.000001))
        # c_att_idx=(channel_att_norm>0.5).nonzero().view(-1)#(channel_att_old>0.5).nonzero().view(-1)




        # c_att_idx_low = (channel_att_norm < 0.2).nonzero().view(-1)
        # c_att_idx_high = (channel_att_norm > 0.8).nonzero().view(-1)
        # base_feat_high = base_feat[:,c_att_idx_high.view(-1).long()]
        # base_feat_high = base_feat_high.reshape(base_feat_high.shape[0],base_feat_high.shape[1],-1)
        # base_feat_org_high = base_feat_org[:,c_att_idx_high.view(-1).long()]
        # base_feat_org_high = base_feat_org_high.reshape(base_feat_org_high.shape[0], base_feat_org_high.shape[1], -1)
        # base_feat_low = base_feat[:, c_att_idx_low.view(-1).long()]
        # base_feat_low = base_feat_low.reshape(base_feat_low.shape[0], base_feat_low.shape[1], -1)
        # base_feat_org_low = base_feat_org[:, c_att_idx_low.view(-1).long()]
        # base_feat_org_low = base_feat_org_low.reshape(base_feat_org_low.shape[0], base_feat_org_low.shape[1], -1)
        # # base_feat_corr_loss=torch.Tensor([0]).cuda()
        # # for i in range(0, base_feat.shape[0]):
        # #     fea_norm_org_low = F.normalize(base_feat_org_low[i], dim=1)### dim=0 20210222ydb
        # #     fea_norm_org_high = F.normalize(base_feat_org_high[i], dim=1)### dim=0 20210222ydb
        # #     fea_norm_low = F.normalize(base_feat_low[i], dim=1)### dim=0 20210222ydb
        # #     fea_norm_high = F.normalize(base_feat_high[i], dim=1)### dim=0 20210222ydb
        # #     sim_matrix_org = fea_norm_org_high.mm(fea_norm_org_low.t())
        # #     sim_matrix = fea_norm_high.mm(fea_norm_low.t())
        # #     base_feat_corr_loss += l1_loss_fn(sim_matrix, sim_matrix_org)
        # base_feat_distil_loss = correlation_distillation_loss(base_feat_low,base_feat_high,base_feat_org_low,base_feat_org_high)\
        # #                      + l1_loss_fn(c_att_org.view(-1), c_att.view(-1))
        #




        #base_feat_distil_loss = l1_loss_fn(base_feat[:,c_att_idx.view(-1).long()], base_feat_org[:,c_att_idx.view(-1).long()])\
        #                         #+l1_loss_fn(c_att_org.view(-1),c_att.view(-1))
        # base_feat_distil_loss = l1_loss_fn( torch.norm(torch.mean(base_feat[:, c_att_idx.view(-1).long()].squeeze(dim=0),dim=0),p=2, keepdim=True),
        #                                     torch.norm(torch.mean(base_feat_org[:, c_att_idx.view(-1).long()].squeeze(dim=0),dim=0),p=2, keepdim=True)) \
        #                         + l1_loss_fn(c_att_org.view(-1), c_att.view(-1))
        # base_feat_distil_loss = l1_loss_fn(base_feat, base_feat_org)
        #base_feat_distil_loss = l1_loss_fn(base_feat*channel_att_norm.view(base_feat.shape[0],base_feat.shape[1],1,1).expand_as(base_feat),base_feat_org*channel_att_norm.view(base_feat.shape[0],base_feat.shape[1],1,1).expand_as(base_feat))
        ###################### feature attention distil ####################
        '''
        base_fea_2=base_feat.squeeze(dim=0).mul(base_feat.squeeze(dim=0))
        base_fea_org_2 = base_feat_org.squeeze(dim=0).mul(base_feat_org.squeeze(dim=0))
        base_fea_att_sum_c = torch.sum(torch.sqrt(base_fea_2),dim=0)
        base_fea_org_att_sum_c = torch.sum(torch.sqrt(base_fea_org_2), dim=0)
        '''
        '''
        base_fea_2 = base_feat.squeeze(dim=0)#.mul(base_feat.squeeze(dim=0))
        base_fea_org_2 = base_feat_org.squeeze(dim=0)#.mul(base_feat_org.squeeze(dim=0))
        base_fea_att_sum_c = torch.sum(base_fea_2, dim=0)
        base_fea_org_att_sum_c = torch.sum(base_fea_org_2, dim=0)
        base_fea_norm = base_fea_att_sum_c /base_feat.shape[1] #(base_fea_att_sum_c - torch.min(base_fea_att_sum_c)) / (torch.max(base_fea_att_sum_c) - torch.min(base_fea_att_sum_c))
        base_fea_org_norm = base_fea_org_att_sum_c/base_feat_org.shape[1] #(base_fea_org_att_sum_c - torch.min(base_fea_org_att_sum_c)) / (torch.max(base_fea_org_att_sum_c) - torch.min(base_fea_org_att_sum_c))
        log_base_fea_org_norm = torch.log(base_fea_org_norm)
        base_feat_distil_loss = F.kl_div(log_base_fea_org_norm, base_fea_norm)
        #base_feat_distil_loss = l2_loss_fn(base_fea_norm, base_fea_org_norm)
        '''

        base_fea_2 = base_feat.squeeze(dim=0)#.mul(base_feat.squeeze(dim=0))
        base_fea_org_2 = base_feat_org.squeeze(dim=0)#.mul(base_feat_org.squeeze(dim=0))
        base_fea_att_sum_c = torch.mean(base_fea_2, dim=0) #/base_feat.shape[1]
        base_fea_org_att_sum_c = torch.mean(base_fea_org_2, dim=0) #/base_feat_org.shape[1]
        base_fea_norm = torch.norm(base_fea_att_sum_c, p=2, keepdim=True)
        base_fea_org_norm = torch.norm(base_fea_org_att_sum_c, p=2, keepdim=True)
        #base_fea_norm = base_fea_att_sum_c/torch.norm(base_fea_att_sum_c, p=2, keepdim=True)
        #base_fea_org_norm = base_fea_org_att_sum_c/torch.norm(base_fea_org_att_sum_c, p=2, keepdim=True)
        l1_loss_fn = torch.nn.L1Loss(reduce=True, size_average=True)
        #base_feat_distil_loss = l1_loss_fn(base_fea_norm, base_fea_org_norm)#(base_feat,base_feat_org)# ablation (base_feat,base_feat_org)#
        base_feat_distil_loss = l1_loss_fn(base_feat,base_feat_org)




















        ''' ablation
        att = torch.mean(torch.abs(base_feat_org), dim=1)  # ,_ max
        att = att.view(base_feat.shape[0], -1)
        min_v, _ = torch.min(att, dim=1)
        max_v, _ = torch.max(att, dim=1)
        att_norm = ((att - min_v.unsqueeze(dim=1).repeat(1, att.shape[1])) / (max_v - min_v + 0.000001).unsqueeze(
            dim=1).repeat(1, att.shape[
            1]))  # .view(-1)#.view(rpn_conv1_org.shape[0],rpn_conv1_org.shape[2],rpn_conv1_org.shape[3]) ## transfer the high response points relation to low response points
        high_point = 0.8  # 0.8
        low_point = 0.1  # 0.1
        high_idx = (att_norm > high_point).nonzero()  # .view(-1)
        low_idx = (att_norm < low_point).nonzero()
        att_new = torch.mean(torch.abs(base_feat), dim=1)  # , _ max
        att_new = att_new.view(base_feat.shape[0], -1)
        min_v_new, _ = torch.min(att_new, dim=1)
        max_v_new, _ = torch.max(att_new, dim=1)
        att_norm_new = ((att_new - min_v_new.unsqueeze(dim=1).repeat(1, att_new.shape[1])) / (
                    max_v_new - min_v_new + 0.000001).unsqueeze(dim=1).repeat(1, att_new.shape[
            1]))  # .view(-1)  # .view(rpn_conv1_org.shape[0],rpn_conv1_org.shape[2],rpn_conv1_org.shape[3]) ## transfer the high response points relation to low response points

        # low_idx = low_idx[random.sample(range(0, len(low_idx)), 300)]
        att_norm_new_high = att_norm_new[high_idx[:, 0].long(), high_idx[:, 1].long()]  # [high_idx.long()]
        high_org = base_feat_org.view(base_feat_org.shape[0], base_feat_org.shape[1], -1)[:, :,
                   high_idx[:, 1].long()]  # [high_idx[:,0].long(), :, high_idx[:,1].long()]#[:, :, high_idx.long()]
        low_org = base_feat_org.view(base_feat_org.shape[0], base_feat_org.shape[1], -1)[:, :,
                  low_idx[:, 1].long()]  # [low_idx[:,0].long(), :, low_idx[:,1].long()]#[:, :, low_idx.long()]
        high_new = base_feat.view(base_feat.shape[0], base_feat.shape[1], -1)[:, :,
                   high_idx[:, 1].long()]  # [high_idx[:,0].long(), :, high_idx[:,1].long()]#[:, :, high_idx.long()]
        low_new = base_feat.view(base_feat.shape[0], base_feat.shape[1], -1)[:, :,
                  low_idx[:, 1].long()]  # [low_idx[:,0].long(), :, low_idx[:,1].long()]#[:, :, low_idx.long()]
        high_org_norm = F.normalize(high_org, dim=1)
        low_org_norm = F.normalize(low_org, dim=1)
        high_new_norm = F.normalize(high_new, dim=1)
        low_new_norm = F.normalize(low_new, dim=1)
        sim_matrix_org = torch.zeros(high_org_norm.shape[0], high_org_norm.shape[2], low_org_norm.shape[2]).cuda()
        sim_matrix_new = torch.zeros(high_new_norm.shape[0], high_new_norm.shape[2], low_new_norm.shape[2]).cuda()
        # if torch.cuda.is_available():
        #     sim_matrix_new=sim_matrix_new.cuda()
        #     sim_matrix_org=sim_matrix_org.cuda()
        for b in range(high_org_norm.shape[0]):
            sim_matrix_org[b] = high_org_norm[b].t().mm(low_org_norm[b])
        for b in range(high_new_norm.shape[0]):
            sim_matrix_new[b] = high_new_norm[b].t().mm(low_new_norm[b])
        base_feat_distil_loss = l1_loss_fn(F.normalize(base_feat,dim=1),F.normalize(base_feat_org,dim=1))#l1_loss_fn(high_new_norm, high_org_norm)+l1_loss_fn(low_new_norm, low_org_norm)#torch.norm((sim_matrix_org - sim_matrix_new).mul(sim_matrix_org - sim_matrix_new))
        '''

















        #####################################################################

        # feed base feature map tp RPN to obtain rois
        #rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
        if bg_coords is not None:
            gt_boxes_bg = torch.cat((gt_boxes[:,:num_boxes[0]], bg_coords.cuda().float()), dim=1)
            num_boxes_bg = num_boxes + bg_coords.shape[1]
        else:
            gt_boxes_bg = gt_boxes
            num_boxes_bg = num_boxes
        rois_, rpn_loss_cls, rpn_loss_bbox, att_loss_ = self.RCNN_rpn(base_feat, im_info, gt_boxes_bg, num_boxes_bg, fasterRCNN_org=fasterRCNN_org, base_feat_ori=base_feat_org)  #### rpn original
        rois, rpn_loss_cls_, rpn_loss_bbox_, att_loss = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes, fasterRCNN_org=fasterRCNN_org, base_feat_ori=base_feat_org)  #### rpn original

        #rois, rpn_loss_cls, rpn_loss_bbox,rpn_conv1_distil_loss= self.RCNN_rpn(base_feat_add, im_info, gt_boxes, num_boxes,fasterRCNN_org)### rpn conv optional
        #rois, rpn_loss_cls, rpn_loss_bbox,rpn_conv1_distil_loss= self.RCNN_rpn(base_feat_add, im_info, gt_boxes, num_boxes,\
        #             fasterRCNN_org,fasterRCNN_residual,base_feat_org=base_feat_org,base_feat_residual=base_feat_res,base_feat_inc=base_feat)### rpn residual conv optional

        # rois_r, rpn_loss_cls_r, rpn_loss_bbox_r = fasterRCNN_residual.RCNN_rpn(base_feat_res, im_info, gt_boxes_n, num_boxes_n)

        #rois_r_c, rpn_loss_cls_r_c, rpn_loss_bbox_r_c = fasterRCNN_residual.RCNN_rpn(base_feat_res_c, im_info, gt_boxes_n,
        #                                                                      num_boxes_n)
        #rpn_loss_cls_r+=rpn_loss_cls_r_c
        #rpn_loss_bbox_r+=rpn_loss_bbox_r_c

        '''
        ################## RPN embedding loss #################################
        l2_loss_fn_sum = torch.nn.MSELoss(reduce=True, size_average=True)#, reduction = 'sum')
        old_rpn_embed = torch.cat((fasterRCNN_org.RCNN_rpn.RPN_cls_score._parameters['weight'].data.view(18,-1),
                                   fasterRCNN_org.RCNN_rpn.RPN_cls_score._parameters['bias'].data.view(18,-1)), dim=1)
        old_fg = torch.mean(old_rpn_embed[::2],dim=0)
        old_bg = torch.mean(old_rpn_embed[1::2],dim=0)
        old_fg_norm = torch.norm(old_fg, p=2, keepdim=True)
        old_bg_norm = torch.norm(old_bg, p=2, keepdim=True)
        new_rpn_embed = torch.cat((self.RCNN_rpn.RPN_cls_score._parameters['weight'].data.view(18,-1),
                                   self.RCNN_rpn.RPN_cls_score._parameters['bias'].data.view(18,-1)), dim=1)
        new_fg = torch.mean(new_rpn_embed[::2],dim=0)
        new_bg = torch.mean(new_rpn_embed[1::2],dim=0)
        new_fg_norm = torch.norm(new_fg, p=2, keepdim=True)
        new_bg_norm = torch.norm(new_bg, p=2, keepdim=True)
        #rpn_embed_distil_loss = l2_loss_fn_sum(new_rpn_embed,old_rpn_embed)/18
        #rpn_embed_distil_loss = l2_loss_fn_sum(new_fg,old_fg)+l2_loss_fn_sum(new_bg,old_bg)
        rpn_embed_distil_loss = l2_loss_fn(new_fg_norm, old_fg_norm) + l2_loss_fn(new_bg_norm, old_bg_norm)
        #######################################################################
        
        rpn_embed_distil_loss=rpn_conv1_distil_loss#torch.Tensor([0]).cuda()
        '''

        #rois, rpn_loss_cls, rpn_loss_bbox, \
        #    =self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes, fasterRCNN_org=fasterRCNN_org) ########### rpn_distil
        #rpn_cls_distil_loss, rpn_bbox_distil_loss

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))

            # roi_data_r = fasterRCNN_residual.RCNN_proposal_target(rois_r, gt_boxes_n, num_boxes_n)
            # rois_r, rois_label_r, rois_target_r, rois_inside_ws_r, rois_outside_ws_r = roi_data_r
            # rois_label_r = Variable(rois_label_r.view(-1).long())
            # rois_target_r = Variable(rois_target_r.view(-1, rois_target_r.size(2)))
            # rois_inside_ws_r = Variable(rois_inside_ws_r.view(-1, rois_inside_ws_r.size(2)))
            # rois_outside_ws_r = Variable(rois_outside_ws_r.view(-1, rois_outside_ws_r.size(2)))

            gt_boxes_roi = gt_boxes.new(gt_boxes.size()).zero_()
            gt_boxes_roi[:, :, 1:5] = gt_boxes[:, :, :4]
            rois_gt = gt_boxes_roi[:, :num_boxes]
            rois_label_gt = Variable(gt_boxes[:, :num_boxes, -1].view(-1).long())
            if bg_coords is not None:
                gt_boxes_roi_bg = gt_boxes_bg.new(gt_boxes_bg.size()).zero_()
                gt_boxes_roi_bg[:, :, 1:5] = gt_boxes_bg[:, :, :4]
                rois_label_gt_bg = Variable(gt_boxes_bg[:, :num_boxes_bg, -1].view(-1).long())
                rois_label_gt_bg_old_idx = ((rois_label_gt_bg<=(self.n_classes - self.n_new_class - 1)) & (rois_label_gt_bg>0)).nonzero().view(-1)
                rois_label_gt_bg_mix_idx = (rois_label_gt_bg<0).nonzero().view(-1)

                mix_relation_loss = torch.Tensor([0]).cuda()
                if rois_label_gt_bg_old_idx.shape[0]>1 and rois_label_gt_bg_mix_idx.shape[0]>1:
                    rois_label_gt_bg = rois_label_gt_bg#[rois_label_gt_bg_old_idx]
                    rois_gt_bg = gt_boxes_roi_bg[:, :num_boxes_bg]#[:, rois_label_gt_bg_old_idx]
                    if cfg.POOLING_MODE == 'align':
                        pooled_feat_gt_bg = self.RCNN_roi_align(base_feat, rois_gt_bg.view(-1, 5))
                        pooled_feat_org_gt_bg = fasterRCNN_org.RCNN_roi_align(base_feat_org, rois_gt_bg.view(-1, 5))  # self
                    elif cfg.POOLING_MODE == 'pool':
                        pooled_feat_gt_bg = self.RCNN_roi_pool(base_feat, rois_gt_bg.view(-1, 5))
                        pooled_feat_org_gt_bg = fasterRCNN_org.RCNN_roi_pool(base_feat_org, rois_gt_bg.view(-1, 5))  # self
                    pooled_feat_gt_bg = self._head_to_tail(pooled_feat_org_gt_bg)
                    pooled_feat_org_gt_bg = fasterRCNN_org._head_to_tail(pooled_feat_org_gt_bg)
                    pooled_feat_gt_bg = pooled_feat_gt_bg.mean(3).mean(2)
                    pooled_feat_org_gt_bg = pooled_feat_org_gt_bg.mean(3).mean(2)
                    pooled_feat_gt_bg_1 = pooled_feat_gt_bg[rois_label_gt_bg_old_idx]
                    pooled_feat_org_gt_bg_1 = pooled_feat_org_gt_bg[rois_label_gt_bg_old_idx]
                    pooled_feat_gt_bg_2 = pooled_feat_gt_bg[rois_label_gt_bg_mix_idx]
                    pooled_feat_org_gt_bg_2 = pooled_feat_org_gt_bg[rois_label_gt_bg_mix_idx]
                    # mix_relation_loss = correlation_distillation_loss_twodim(pooled_feat_gt_bg, pooled_feat_org_gt_bg)
                    mix_relation_loss = correlation_distillation_loss_twodim_2(pooled_feat_gt_bg_1, pooled_feat_gt_bg_2, pooled_feat_org_gt_bg_1, pooled_feat_org_gt_bg_2)

        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

            # rois_label_r = None
            # rois_target_r = None
            # rois_inside_ws_r = None
            # rois_outside_ws_r = None
            # rpn_loss_cls_r = 0
            # rpn_loss_bbox_r = 0

        rois = Variable(rois)
        # rois_r = Variable(rois_r)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
            #pooled_feat_inc = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
            pooled_feat_org = fasterRCNN_org.RCNN_roi_align(base_feat_org, rois.view(-1, 5))#self
            # pooled_feat_r = fasterRCNN_residual.RCNN_roi_align(base_feat_res, rois_r.view(-1, 5))
            # pooled_feat_r_roiinc=fasterRCNN_residual.RCNN_roi_align(base_feat_res, rois.view(-1, 5))

            #pooled_feat_org_roi_r=fasterRCNN_org.RCNN_roi_align(base_feat_org, rois_r.view(-1, 5))
            #pooled_feat_roi_r = self.RCNN_roi_align(base_feat_add, rois_r.view(-1, 5))
            #pooled_feat_r_copy = fasterRCNN_residual.RCNN_roi_align(base_feat_res_copy, rois_r.view(-1, 5))


        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))
            #pooled_feat_inc = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))
            pooled_feat_org = fasterRCNN_org.RCNN_roi_pool(base_feat_org, rois.view(-1, 5))#self
            # pooled_feat_r = fasterRCNN_residual.RCNN_roi_pool(base_feat_res, rois_r.view(-1, 5))
            # pooled_feat_r_roiinc = fasterRCNN_residual.RCNN_roi_pool(base_feat_res, rois.view(-1, 5))

            #pooled_feat_org_roi_r = fasterRCNN_org.RCNN_roi_pool(base_feat_org, rois_r.view(-1, 5))
            #pooled_feat_roi_r = self.RCNN_roi_pool(base_feat_add, rois_r.view(-1, 5))
            #pooled_feat_r_copy = fasterRCNN_residual.RCNN_roi_pool(base_feat_res_copy, rois_r.view(-1, 5))

        #pooled_feat_distil_loss_1 = l1_loss_fn(pooled_feat - pooled_feat_r_roiinc, pooled_feat_org) + l1_loss_fn(
        #    pooled_feat - pooled_feat_org, pooled_feat_r_roiinc)

        idx_fgold = ((rois_label > 0) & (rois_label <= self.n_classes - self.n_new_class - 1)).nonzero().view(-1)
        if idx_fgold.shape[0] == 0:
            l1_pooled_fgold_loss = torch.Tensor([0]).cuda()
        else:
            l1_pooled_fgold_loss = l1_loss_fn(pooled_feat[idx_fgold], pooled_feat_org[idx_fgold])

        # feed pooled features to top model

        isda_loss = torch.Tensor([0]).cuda()
        if self.training is True and isda_criterion is not None and cfg.TRAIN.pool:# and cfg.TRAIN.feadim!=2048:
            ratio = 0.5 * (epoch / 10)#0.5 #* (epoch / (training_configurations[args.model]['epochs']))
            class_fea = []
            class_target = []
            all =  True
            use_gt = False#True#
            if all:
                if use_gt:
                    idx_roi = ((rois_label_gt > 0) ).nonzero().view(-1)
                    idx_roi_old = ((rois_label_gt > 0) & (rois_label_gt <= self.n_classes - self.n_new_class - 1)).nonzero().view(-1)
                else:
                    idx_roi = ((rois_label > 0) ).nonzero().view(-1)
                    idx_roi_old = ((rois_label > 0) & (rois_label <= self.n_classes - self.n_new_class - 1)).nonzero().view(-1)
            else:
                if use_gt:
                    idx_roi = ((rois_label_gt > 0) & (rois_label_gt <= self.n_classes - self.n_new_class - 1)).nonzero().view(-1)
                else:
                    idx_roi = ((rois_label > 0) & (rois_label <= self.n_classes - self.n_new_class - 1)).nonzero().view(-1)
                idx_roi_old = idx_roi
            if use_gt:
                pooled_feat_gt = self.RCNN_roi_pool(base_feat, rois_gt.view(-1, 5))
                pooled_feat_org_gt = fasterRCNN_org.RCNN_roi_pool(base_feat_org, rois_gt.view(-1, 5))
                # pooled_feat_gt = self._head_to_tail(pooled_feat_gt)
                # pooled_feat_org_gt = fasterRCNN_org._head_to_tail(pooled_feat_org_gt)
                if self.extra_fea_pool is not None:
                    pooled_feat_gt_ex = self.extra_fea_pool(pooled_feat_gt)#pooled_feat_gt#
                    pooled_feat_org_gt_ex = self.extra_fea_pool(pooled_feat_org_gt)#pooled_feat_org_gt#
                    pooled_feat_gt_ex = pooled_feat_gt_ex.mean(3).mean(2)
                    pooled_feat_org_gt_ex = pooled_feat_org_gt_ex.mean(3).mean(2)
                else:
                    pooled_feat_gt_ex = pooled_feat_gt.mean(3).mean(2)
                    pooled_feat_org_gt_ex = pooled_feat_org_gt.mean(3).mean(2)
                # pooled_feat_gt = self.extra_fea_pool(pooled_feat_gt)
                # pooled_feat_org_gt = self.extra_fea_pool(pooled_feat_org_gt)
                if idx_roi.shape[0]>0:
                    class_fea_old = pooled_feat_org_gt_ex[idx_roi]
                    class_target = rois_label_gt[idx_roi]
                    class_fea_new = pooled_feat_gt_ex[idx_roi]
                    if epoch<0:
                        class_fea = class_fea_old#torch.cat((class_fea_old, class_fea_new),dim=0)#class_fea_old#
                        class_target = class_target#torch.cat((class_target, class_target),dim=0)#class_target#
                    else:
                        class_fea = class_fea_old #new#torch.cat((class_fea_old, class_fea_new),dim=0)#class_fea_old#
                        class_target = class_target#torch.cat((class_target, class_target),dim=0)#class_target#
            else:
                if self.extra_fea_pool is not None:
                    pooled_feat_ex = self.extra_fea_pool(pooled_feat)  # pooled_feat_gt#
                    pooled_feat_org_ex = self.extra_fea_pool(pooled_feat_org)  # pooled_feat_org_gt#
                    pooled_feat_ex = pooled_feat_ex.mean(3).mean(2)
                    pooled_feat_org_ex = pooled_feat_org_ex.mean(3).mean(2)
                else:
                    pooled_feat_ex = pooled_feat.mean(3).mean(2)
                    pooled_feat_org_ex = pooled_feat_org.mean(3).mean(2)
                if idx_roi.shape[0] > 0:
                    class_fea_old = pooled_feat_org_ex[idx_roi]
                    class_fea_new = pooled_feat_ex[idx_roi]
                    class_target = rois_label[idx_roi]
                    class_fea = class_fea_old #torch.cat((class_fea_old, class_fea_new),dim=0)
                    # # class_target = torch.cat((class_target, class_target),dim=0)
                    # class_fea = self.extra_fea_pool(class_fea)

            if len(class_fea)>0 and len(class_target)>0:
                if cfg.TRAIN.rdc:
                    class_fea = F.avg_pool1d(class_fea.unsqueeze(dim=1), kernel_size=4, stride=4).squeeze(dim=1)
                # if epoch>6:
                #     self.extra_classifier.requires_grad = False
                # isda_loss, isda_output = isda_criterion(class_fea, (self.RCNN_cls_score, self.RCNN_cls_score_new), class_fea,
                #                                     class_target, ratio)
                extra_logit = self.extra_classifier(class_fea)


                ce_loss = F.cross_entropy(extra_logit, class_target-1)
                # isda_loss, isda_output = isda_criterion(class_fea, self.extra_classifier, class_fea, class_target-1, ratio)

                kg = isda_tmp[0]
                out_new = isda_tmp[1]
                feature_mean = isda_tmp[2]
                isda_loss, isda_output = isda_criterion(class_fea, self.extra_classifier, class_fea, class_target-1, ratio, kg=kg, out_new=out_new, feature_mean=feature_mean)
                # isda_loss, isda_output = isda_criterion(class_fea, self.extra_classifier, (fasterRCNN_org.RCNN_cls_score, self.RCNN_cls_score_new), class_fea, class_target-1, ratio, kg=kg, out_new=out_new, feature_mean=feature_mean)

                # isda_loss = ce_loss
                # cv_var = isda_criterion.get_cv().cuda()
                # cv_var.requried_grad=True
                # isda_criterion(self.extra_classifier, class_fea, torch.argmax(extra_logit, dim=1)[0], class_target-1, alpha, weights, cv_var, "update", kg, out_new,
                #  feature_mean, beta, args.head)

                # isda_loss += ce_loss
                # isda_loss, isda_output = isda_criterion(class_fea, self.RCNN_cls_score,
                #                                         class_fea,
                #                                         class_target, ratio)
                if use_gt:
                    pooled_feat_gt = self._head_to_tail(pooled_feat_gt)
                    pooled_feat_org_gt = fasterRCNN_org._head_to_tail(pooled_feat_org_gt)
                    pooled_feat_gt = pooled_feat_gt.mean(3).mean(2)
                    pooled_feat_org_gt = pooled_feat_org_gt.mean(3).mean(2)
                    rcnn_logit = self.RCNN_cls_score(pooled_feat_gt[idx_roi])
                    # rcnn_logit_new = self.RCNN_cls_score_new(pooled_feat_gt[idx_roi])
                    # rcnn_logit = torch.cat((rcnn_logit, rcnn_logit_new),dim=1)
                    rcnn_logit_org = fasterRCNN_org.RCNN_cls_score(pooled_feat_org_gt[idx_roi])
                else:
                    pooled_feat_ngt = self._head_to_tail(pooled_feat)
                    pooled_feat_org_ngt = fasterRCNN_org._head_to_tail(pooled_feat_org)
                    pooled_feat_ngt = pooled_feat_ngt.mean(3).mean(2)
                    pooled_feat_org_ngt = pooled_feat_org_ngt.mean(3).mean(2)
                    rcnn_logit = self.RCNN_cls_score(pooled_feat_ngt[idx_roi])
                    rcnn_logit_org = fasterRCNN_org.RCNN_cls_score(pooled_feat_org_ngt[idx_roi])

                logit_loss = torch.tensor(0.0).cuda()

                logp_x = F.log_softmax(extra_logit[:,:rcnn_logit_org[:,1:].shape[1]], 1)
                p_y = F.softmax(rcnn_logit_org[:,1:], 1)
                # logp_x = F.log_softmax(rcnn_logit[:,1:], 1)
                # p_y = F.softmax(extra_logit[:,:rcnn_logit_org[:,1:].shape[1]], 1)
                if step%100==0:
                    print('logit_shape:',extra_logit[:,:rcnn_logit_org[:,1:].shape[1]].shape,rcnn_logit_org[:,1:].shape)
                logit_loss = F.l1_loss(extra_logit[:,:rcnn_logit_org[:,1:].shape[1]], rcnn_logit_org[:,1:]) #F.kl_div(logp_x, p_y)# F.l1_loss(rcnn_logit[:,1:], extra_logit)+

                old_new_logit_loss = torch.tensor(0.0).cuda()
                # if idx_roi_old.shape[0]>0:
                #     extra_logit_old = self.extra_classifier(class_fea_old[idx_roi_old])
                #     extra_logit_new = self.extra_classifier(class_fea_new[idx_roi_old])
                #     old_new_logit_loss = F.l1_loss(extra_logit_new[:, :rcnn_logit_org[:, 1:].shape[1]],
                #                                extra_logit_old[:, :rcnn_logit_org[:, 1:].shape[1]]) \
                #                         + F.l1_loss(class_fea_new[idx_roi_old], class_fea_old[idx_roi_old])
                # print(logit_loss, old_new_logit_loss, isda_loss)
                isda_loss += (logit_loss + old_new_logit_loss)
                if exsup:
                    if self.extra_fea_pool is not None:
                        pooled_feat_exsup = self.extra_fea_pool(pooled_feat)  # pooled_feat_gt#
                        pooled_feat_org_exsup = self.extra_fea_pool(pooled_feat_org)  # pooled_feat_org_gt#
                        pooled_feat_exsup = pooled_feat_exsup.mean(3).mean(2)
                        pooled_feat_org_exsup = pooled_feat_org_exsup.mean(3).mean(2)
                    else:
                        pooled_feat_exsup = pooled_feat.mean(3).mean(2)
                        pooled_feat_org_exsup = pooled_feat_org.mean(3).mean(2)
                    if cfg.TRAIN.rdc:
                        pooled_feat_org_exsup = F.avg_pool1d(pooled_feat_org_exsup.unsqueeze(dim=1), kernel_size=4, stride=4).squeeze(dim=1)
                    extra_logit_exsup = self.extra_classifier(pooled_feat_org_exsup)

                if step%100==0:
                    print('ce_loss: %.4f, old_new_logit_loss: %.4f, logit_loss: %.4f' % (ce_loss, old_new_logit_loss, logit_loss))
            else:
                extra_logit_exsup = None

        mixup_loss = torch.Tensor([0]).cuda()
        if cfg.TRAIN.mixup:
            mixup_dsl = torch.tensor(0.0).cuda()
            mixup_cr = nn.CrossEntropyLoss()
            if not imglevel:
                idx_roi_mix = ((rois_label > 0)).nonzero().view(-1)
                idx_roi_mix = ((rois_label > 0) & (rois_label <= self.n_classes - self.n_new_class - 1)).nonzero().view(-1)
                if idx_roi_mix.shape[0]>0:
                    mixup_inputs = pooled_feat[idx_roi_mix]
                    mixup_inputs_old = pooled_feat_org[idx_roi_mix]
                    mixup_targets = rois_label[idx_roi_mix]
                else:
                    mixup_inputs, mixup_targets_a, mixup_targets_b, mixup_lam = None, None, None, None
                if mixup_inputs is not None:
                    mixup_inputs, mixup_targets_a, mixup_targets_b, mixup_lam = mixup_data_old(mixup_inputs, mixup_targets,
                                                                                               mixup_inputs_old)
                    # mixup_inputs, mixup_targets_a, mixup_targets_b, mixup_lam = mixup_data(mixup_inputs, mixup_targets)

            else:
                mix_rois_gt = Variable(mix_rois_gt)
                # do roi pooling based on predicted rois
                if cfg.POOLING_MODE == 'align':
                    mix_pooled_feat = self.RCNN_roi_align(mix_base_feat, mix_rois_gt[:,:,:5].view(-1, 5))
                    mix_pooled_feat_org = fasterRCNN_org.RCNN_roi_align(mix_base_feat_org, mix_rois_gt[:,:,:5].view(-1, 5))
                elif cfg.POOLING_MODE == 'pool':
                    mix_pooled_feat = self.RCNN_roi_pool(mix_base_feat, mix_rois_gt[:,:,:5].view(-1, 5))
                    mix_pooled_feat_org = fasterRCNN_org.RCNN_roi_pool(mix_base_feat_org, mix_rois_gt[:,:,:5].view(-1, 5))


                mixup_inputs = mix_pooled_feat
                mixup_targets_a = mix_rois_label_gt_a
                mixup_targets_b = mix_rois_label_gt_b
                a_old = ((mixup_targets_a > 0) & (mixup_targets_a <= self.n_classes - self.n_new_class - 1)).nonzero().view(-1)
                b_old = ((mixup_targets_b > 0) & (mixup_targets_b <= self.n_classes - self.n_new_class - 1)).nonzero().view(-1)
                ab_old = ((mixup_targets_a > 0) & (mixup_targets_a <= self.n_classes - self.n_new_class - 1) & (mixup_targets_b > 0) & (mixup_targets_b <= self.n_classes - self.n_new_class - 1)).nonzero().view(-1)

                # if a_old.shape[0]>0:
                #     mixup_dsl += mixup_lam * F.l1_loss(mix_pooled_feat[a_old], mix_pooled_feat_org[a_old])
                # if b_old.shape[0]>0:
                #     mixup_dsl += (1-mixup_lam) * F.l1_loss(mix_pooled_feat[b_old], mix_pooled_feat_org[b_old])
                if ab_old.shape[0]>0:
                    mix_pooled_feat_ab_old = self._head_to_tail(mix_pooled_feat[ab_old])
                    mix_pooled_feat_ab_old = mix_pooled_feat_ab_old.mean(3).mean(2)
                    mixup_targets_a_old = mixup_targets_a[ab_old]
                    mixup_targets_b_old = mixup_targets_b[ab_old]
                    mixup_dsl += (mixup_lam*F.l1_loss(F.normalize(mix_pooled_feat_ab_old, p=2,dim=-1), F.normalize(prototypes[mixup_targets_a_old-1], p=2,dim=-1))+(1-mixup_lam)*F.l1_loss(F.normalize(mix_pooled_feat_ab_old, p=2,dim=-1), F.normalize(prototypes[mixup_targets_b_old-1], p=2,dim=-1)))
                    # mixup_dsl += F.l1_loss(mix_pooled_feat[ab_old], mix_pooled_feat_org[ab_old])
                if step%100==0:
                    print('mixup_dsl %.4f' % (mixup_dsl.item()))

            mixup_outputs = self._head_to_tail(mixup_inputs)
            mixup_outputs = mixup_outputs.mean(3).mean(2)

            # compute object classification probability
            cls_score_mix = self.RCNN_cls_score(mixup_outputs)
            cls_prob_mix = F.softmax(cls_score_mix, 1)

            ################# split score fc (old and new)#####################
            cls_score_new_mix = self.RCNN_cls_score_new(mixup_outputs)
            cls_score_cat_mix = torch.cat((cls_score_mix, cls_score_new_mix), dim=1)
            cls_prob_mix = F.softmax(cls_score_cat_mix, 1)
            mixup_outputs = cls_score_cat_mix
            ###################################################################
            if self.training:
                # mixup_loss = mixup_criterion(mixup_cr, mixup_outputs, mixup_targets_a, mixup_targets_b, mixup_lam)
                # mixup_loss = mixup_criterion_kl(mixup_outputs, mixup_targets_a, mixup_targets_b, mixup_lam)
                mixup_loss = mixup_dsl

        pooled_feat = self._head_to_tail(pooled_feat)#(256,1024,7,7)->(256,2048,4,4)
        pooled_feat_org = fasterRCNN_org._head_to_tail(pooled_feat_org)  # self


        rois_fg=rois_label.ne(0).nonzero().view(-1)#((rois_label>0) & (rois_label<(self.n_classes-self.n_new_class))).nonzero().view(-1)#
        if rois_fg.shape[0]>0:
            rois_fg_fea= pooled_feat[rois_fg.view(-1).long()]
            rois_fg_fea_org = pooled_feat_org[rois_fg.view(-1).long()]
            k=2
            width=rois_fg_fea.shape[2]
            height=rois_fg_fea.shape[3]
            sub_width=int(width/k)
            sub_height=int(height/k)

            sim_matrix_intra_instance = torch.zeros(rois_fg_fea.shape[0], k*k, k*k)
            sim_matrix_intra_instance_org = torch.zeros(rois_fg_fea.shape[0], k*k, k*k)
            if torch.cuda.is_available():
                sim_matrix_intra_instance_org=sim_matrix_intra_instance_org.cuda()
                sim_matrix_intra_instance=sim_matrix_intra_instance.cuda()
            # rois_fg_fea=rois_fg_fea.reshape(rois_fg_fea.shape[0], rois_fg_fea.shape[1], 16).permute(0, 2, 1)
            # rois_fg_fea_org=rois_fg_fea_org.reshape(rois_fg_fea_org.shape[0],rois_fg_fea_org.shape[1],16).permute(0,2,1)
            #
            #
            # for b in range(rois_fg_fea.shape[0]):
            #     fea_norm=F.normalize(rois_fg_fea[b],dim=1)### dim=0 20210222ydb
            #     fea_norm_org=F.normalize(rois_fg_fea_org[b],dim=1)### dim=0 20210222ydb
            #     sim_matrix_intra_instance[b]=fea_norm.mm(fea_norm.t())
            #     sim_matrix_intra_instance_org[b] = fea_norm_org.mm(fea_norm_org.t())

            # for b in range(rois_fg_fea.shape[0]):
            #     patch_vec = torch.Tensor([])
            #     patch_vec_org = torch.Tensor([])
            #     if torch.cuda.is_available():
            #         patch_vec_org=patch_vec_org.cuda()
            #         patch_vec=patch_vec.cuda()
            #     for i in range(0,k):
            #         for j in range(0,k):
            #             patch = rois_fg_fea[b,:,i*sub_width,j*sub_height].view(-1).unsqueeze(dim=0)
            #             patch_org = rois_fg_fea_org[b, :, i * sub_width, j * sub_height].view(-1).unsqueeze(dim=0)
            #             if patch_vec.shape[0] == 0:
            #                 patch_vec=patch
            #                 patch_vec_org=patch_org
            #             else:
            #                 patch_vec=torch.cat((patch_vec,patch),dim=0)
            #                 patch_vec_org = torch.cat((patch_vec_org,patch_org), dim=0)
            #
            #     fea_norm=F.normalize(patch_vec,dim=1)### dim=0 20210222ydb
            #     fea_norm_org=F.normalize(patch_vec_org,dim=1)### dim=0 20210222ydb
            #     sim_matrix_intra_instance[b]=fea_norm.mm(fea_norm.t())
            #     sim_matrix_intra_instance_org[b] = fea_norm_org.mm(fea_norm_org.t())
            #     # for ii in range(0,k*k):
            #     #     for jj in range(0,k*k):
            #     #         sim_matrix_intra_instance[b][ii][jj] = torch.cosine_similarity(patch_vec[ii].view(-1), patch[jj].view(-1), dim=0)
            #     #         sim_matrix_intra_instance_org[b][ii][jj] = torch.cosine_similarity(patch_org[ii].view(-1), patch_org[jj].view(-1),
            #     #                                                                      dim=0)

            patch_vec = torch.Tensor([])
            patch_vec_org = torch.Tensor([])
            if torch.cuda.is_available():
                patch_vec_org = patch_vec_org.cuda()
                patch_vec = patch_vec.cuda()
            for i in range(0, k):
                for j in range(0, k):
                    patch = torch.mean(rois_fg_fea[:, :, i * sub_width:(i+1)*sub_width, j * sub_height:(j+1)*sub_height].contiguous(),dim=1).view(rois_fg_fea.shape[0],-1).unsqueeze(dim=1)
                    patch_org = torch.mean(rois_fg_fea_org[:, :, i * sub_width:(i+1)*sub_width, j * sub_height:(j+1)*sub_height].contiguous(),dim=1).view(rois_fg_fea_org.shape[0],-1).unsqueeze(dim=1)
                    if patch_vec.shape[0] == 0:
                        patch_vec = patch
                        patch_vec_org = patch_org
                    else:
                        patch_vec = torch.cat((patch_vec, patch), dim=1)
                        patch_vec_org = torch.cat((patch_vec_org, patch_org), dim=1)


            fea_norm = F.normalize(patch_vec, dim=1)
            fea_norm_org = F.normalize(patch_vec_org, dim=1)
            for i in range(0,fea_norm.shape[0]):
                sim_matrix_intra_instance[i] = fea_norm[i].mm(fea_norm[i].t())
                sim_matrix_intra_instance_org[i] = fea_norm_org[i].mm(fea_norm_org[i].t())
            intra_roi_loss=l1_loss_fn(sim_matrix_intra_instance_org,sim_matrix_intra_instance)
        else:
            intra_roi_loss = torch.Tensor([0]).cuda()


        l1_pooled_loss = l1_loss_fn(pooled_feat, pooled_feat_org)
        idx_fgold = ((rois_label > 0) & (rois_label <= self.n_classes - self.n_new_class - 1)).nonzero().view(-1)
        idx_fgnew= ( (rois_label > self.n_classes - self.n_new_class - 1)).nonzero().view(-1)

        if idx_fgold.shape[0]==0:
            print('roigt', rois_label[:num_boxes[0]], gt_boxes[0][:num_boxes[0]])
            l1_pooled_fgold_loss = torch.Tensor([0]).cuda()
            img_save = (im_data[0].permute(1, 2, 0) + torch.from_numpy(cfg.PIXEL_MEANS).cuda().float()).cpu().numpy()
            from PIL import Image
            img_save = Image.fromarray(np.uint8(img_save[:, :, ::-1]))
            img_save.save('imgs_save1/'+str(random.randint(0,100000))+'.jpg')
        else:
            l1_pooled_fgold_loss = l1_loss_fn(pooled_feat[idx_fgold], pooled_feat_org[idx_fgold])



        correlation_loss = torch.Tensor([0])
        correlation_loss = intra_roi_loss #+ inter_roi_loss_fgfg#+ inter_roi_loss_fgbg # #+ inter_roi_loss_bgbg#intra_roi_loss #+

        # correlation_loss = l1_loss_fn(pooled_feat, pooled_feat_org)

        cts_loss = torch.Tensor([0]).cuda()
        if cfg.TRAIN.CTS:
            # features: [bsz, n_views, f_dim]
            # `n_views` is the number of crops from each image
            # better be L2 normalized in f_dim dimension
            views=1#1#2#1
            mean = False
            feacts = False #True
            ins = True
            noold = False
            # if cfg.TRAIN.SET=='coco':
            #     mean=True#False#True #
            if feacts and rois_fg.shape[0] > 0:
                pooled_feat_org_avg = F.avg_pool2d(pooled_feat_org, kernel_size=(2, 2), stride=2)
                pooled_feat_avg = F.avg_pool2d(pooled_feat, kernel_size=(2, 2), stride=2)
                if idx_fgold.shape[0]>0:
                    cts_pool_fea_org_old = pooled_feat_org_avg[idx_fgold].permute(0, 2, 3, 1)
                    cts_pool_fea_org_old = cts_pool_fea_org_old.contiguous().view(-1, 2048)
                    cts_pool_fea_old = pooled_feat_avg[idx_fgold].permute(0, 2, 3, 1)
                    cts_pool_fea_old = cts_pool_fea_old.contiguous().view(-1, 2048)
                    cts_pool_label_old = rois_label[idx_fgold].unsqueeze(dim=1).repeat(1, pooled_feat_org_avg[idx_fgold].shape[2] *
                                                                              pooled_feat_org_avg[idx_fgold].shape[3]).view(-1)
                    # sim_cts_org_old = torch.zeros((idx_fgold.shape[0], idx_fgold.shape[0]))
                    # sim_cts_old = torch.zeros((idx_fgold.shape[0], idx_fgold.shape[0]))
                    # cts_pool_fea_old = cts_pool_fea_old.mm(cts_pool_fea_old.t())
                    # cts_pool_fea_org_old = cts_pool_fea_org_old.mm(cts_pool_fea_org_old.t())
                    # features_old = torch.cat((cts_pool_fea_old, cts_pool_fea_org_old), dim=0)
                    # labels_old = torch.cat((cts_pool_label_old, cts_pool_label_old), dim=0)
                    features_old = torch.cat((cts_pool_fea_old.unsqueeze(dim=1), cts_pool_fea_org_old.unsqueeze(dim=1)), dim=1)
                    labels_old = cts_pool_label_old#torch.cat((cts_pool_label_old, cts_pool_label_old), dim=0)
                if idx_fgnew.shape[0]>0:
                    cts_pool_fea_org_new = pooled_feat_org_avg[idx_fgnew].permute(0, 2, 3, 1)
                    cts_pool_fea_org_new = cts_pool_fea_org_new.contiguous().view(-1,2048)
                    cts_pool_fea_new = pooled_feat_avg[idx_fgnew].permute(0, 2, 3, 1)
                    cts_pool_fea_new= cts_pool_fea_new.contiguous().view(-1, 2048)
                    cts_pool_label_new = rois_label[idx_fgnew].unsqueeze(dim=1).repeat(1, pooled_feat_org_avg[idx_fgnew].shape[2]*pooled_feat_org_avg[rois_fg].shape[3]).view(-1)
                    # features_new = cts_pool_fea_new#torch.cat((cts_pool_fea_org_new, cts_pool_fea_new), dim=0)
                    # labels_new = cts_pool_label_new#torch.cat((cts_pool_label_new, cts_pool_label_new), dim=0)

                    features_new = torch.cat((cts_pool_fea_new.unsqueeze(dim=1), cts_pool_fea_new.unsqueeze(dim=1)), dim=1)
                    labels_new = cts_pool_label_new #torch.cat((cts_pool_label_new, cts_pool_label_new), dim=0)

                if idx_fgold.shape[0]>0 and idx_fgnew.shape[0]>0:
                    features = torch.cat((features_old, features_new), dim=0)
                    labels = torch.cat((labels_old, labels_new), dim=0)
                elif idx_fgnew.shape[0]>0:
                    features = features_new
                    labels = labels_new
                elif idx_fgold.shape[0]>0:
                    features = features_old
                    labels = labels_old
                else:
                    features = torch.Tensor([]).cuda()
                    labels = torch.Tensor([]).cuda()
            elif ins:
                if idx_fgold.shape[0] > 0:
                    cts_pool_fea_org_old = pooled_feat_org[idx_fgold].permute(0, 2, 3, 1) # n, 4, 4, 2048
                    cts_pool_fea_old = pooled_feat[idx_fgold].permute(0, 2, 3, 1) # n, 4, 4, 2048
                    cts_pool_fea_org_old = cts_pool_fea_org_old.contiguous().view(-1, 16, 2048)
                    cts_pool_fea_old = cts_pool_fea_old.contiguous().view(-1, 16, 2048)
                    for idx_ins, _ in enumerate(idx_fgold):
                        ins_fea = torch.cat((cts_pool_fea_old[idx_ins].unsqueeze(dim=1), cts_pool_fea_org_old[idx_ins].unsqueeze(dim=1)),dim=1)
                        cts_loss += criterion(ins_fea)
                    cts_loss = cts_loss / idx_fgold.shape[0]
                features = torch.Tensor([]).cuda()
                labels = torch.Tensor([]).cuda()
                if step % 100 == 0:
                    print('ins-level cts_loss: %.4f' % (cts_loss), ins_fea.shape)
            else:
                if noold:
                    cates = rois_label[(rois_label > 0).nonzero().view(-1)].unique()  # rois_label.unique()#
                    # labels = torch.Tensor([i for i in range(1, prototypes.shape[0] + 1)]).long().cuda()
                    # import copy
                    # proto_labels = copy.deepcopy(labels)
                    features = torch.Tensor([]).cuda()  # prototypes.repeat(1,2).reshape(prototypes.shape[0],2,-1).cuda()#torch.Tensor([]).cuda()#torch.cat((prototypes.unsqueeze(dim=1), aug_gaussian_fea(prototypes, stds, fea_num=1).reshape(prototypes.shape[0],-1).unsqueeze(dim=1)), dim=1)#torch.Tensor([]).cuda()#prototypes.repeat(1,2).reshape(prototypes.shape[0],2,-1)prototypes.unsqueeze(dim=1)#torch.Tensor([]).cuda()#
                    for ca in cates:
                        if ca <= self.n_classes - self.n_new_class - 1:  # ca > 0 and
                            ca_fea = pooled_feat[rois_label.eq(ca).nonzero().view(-1)]
                            # ca_fea_org = pooled_feat_org[rois_label.eq(ca).nonzero().view(-1)]
                            if mean:
                                cat_fea = torch.cat(
                                    (
                                    torch.mean(ca_fea, dim=0).unsqueeze(dim=0), torch.mean(ca_fea, dim=0).unsqueeze(dim=0)),
                                    dim=0).unsqueeze(dim=0)
                                # cat_fea = torch.cat((torch.mean(ca_fea,dim=0).unsqueeze(dim=0),torch.mean(ca_fea_org,dim=0).unsqueeze(dim=0)), dim=0).unsqueeze(dim=1)
                            else:
                                cat_fea = torch.cat((ca_fea.unsqueeze(dim=1), ca_fea.unsqueeze(dim=1)), dim=1)
                        else:
                            ca_fea = pooled_feat[rois_label.eq(ca).nonzero().view(-1)]
                            if mean:
                                cat_fea = torch.cat((
                                    torch.mean(ca_fea, dim=0).unsqueeze(dim=0), torch.mean(ca_fea, dim=0).unsqueeze(dim=0)),
                                    dim=0).unsqueeze(dim=0)
                                # cat_fea = torch.cat((torch.mean(ca_fea,dim=0).unsqueeze(dim=0),torch.mean(ca_fea_org,dim=0).unsqueeze(dim=0)), dim=0).unsqueeze(dim=1)
                            else:
                                cat_fea = torch.cat((ca_fea.unsqueeze(dim=1), ca_fea.unsqueeze(dim=1)), dim=1)
                        if features.shape[0] > 0:
                            features = torch.cat((features, cat_fea), dim=0)
                            if mean:
                                labels = torch.cat((labels, torch.Tensor([ca]).long().cuda()), dim=0)
                                # labels = torch.cat((labels, torch.Tensor([ca]).long().cuda()), dim=0)
                            else:
                                labels = torch.cat(
                                    (labels, torch.Tensor([ca for ica in range(0, cat_fea.shape[0])]).long().cuda()), dim=0)
                        else:
                            features = cat_fea
                            if mean:
                                labels = torch.Tensor([ca]).long().cuda()
                            else:
                                labels = torch.Tensor([ca for ica in range(0, cat_fea.shape[0])]).long().cuda()
                    # cates = rois_label[(rois_label > 0).nonzero().view(-1)].unique() #rois_label.unique()#
                    # # labels = torch.Tensor([i for i in range(1, prototypes.shape[0] + 1)]).long().cuda()
                    # # import copy
                    # # proto_labels = copy.deepcopy(labels)
                    # features = torch.Tensor([]).cuda()  # prototypes.repeat(1,2).reshape(prototypes.shape[0],2,-1).cuda()#torch.Tensor([]).cuda()#torch.cat((prototypes.unsqueeze(dim=1), aug_gaussian_fea(prototypes, stds, fea_num=1).reshape(prototypes.shape[0],-1).unsqueeze(dim=1)), dim=1)#torch.Tensor([]).cuda()#prototypes.repeat(1,2).reshape(prototypes.shape[0],2,-1)prototypes.unsqueeze(dim=1)#torch.Tensor([]).cuda()#
                    # for ca in cates:
                    #     if ca<=self.n_classes-self.n_new_class-1:#ca > 0 and
                    #         ca_fea = pooled_feat[rois_label.eq(ca).nonzero().view(-1)]
                    #         ca_fea_org = pooled_feat_org[rois_label.eq(ca).nonzero().view(-1)]
                    #         if mean:
                    #             cat_fea = torch.mean(ca_fea, dim=0).unsqueeze(dim=0).unsqueeze(dim=0)
                    #             # cat_fea = torch.cat((torch.mean(ca_fea,dim=0).unsqueeze(dim=0),torch.mean(ca_fea_org,dim=0).unsqueeze(dim=0)), dim=0).unsqueeze(dim=1)
                    #         else:
                    #             cat_fea = ca_fea.unsqueeze(dim=1)
                    #     else:
                    #         ca_fea = pooled_feat[rois_label.eq(ca).nonzero().view(-1)]
                    #         if mean:
                    #             cat_fea = torch.mean(ca_fea, dim=0).unsqueeze(dim=0).unsqueeze(dim=0)
                    #             # cat_fea = torch.cat((torch.mean(ca_fea,dim=0).unsqueeze(dim=0),torch.mean(ca_fea_org,dim=0).unsqueeze(dim=0)), dim=0).unsqueeze(dim=1)
                    #         else:
                    #             cat_fea = ca_fea.unsqueeze(dim=1)
                    #     if features.shape[0] > 0:
                    #         features = torch.cat((features, cat_fea), dim=0)
                    #         if mean:
                    #             labels = torch.cat((labels, torch.Tensor([ca]).long().cuda()), dim=0)
                    #             # labels = torch.cat((labels, torch.Tensor([ca]).long().cuda()), dim=0)
                    #         else:
                    #             labels = torch.cat((labels, torch.Tensor([ca for ica in range(0, cat_fea.shape[0])]).long().cuda()), dim=0)
                    #     else:
                    #         features = cat_fea
                    #         if mean:
                    #             labels = torch.Tensor([ca]).long().cuda()
                    #         else:
                    #             labels = torch.Tensor([ca for ica in range(0, cat_fea.shape[0])]).long().cuda()
                elif views==1:
                    cates = rois_label[(rois_label > 0).nonzero().view(-1)].unique() #rois_label.unique()#
                    # labels = torch.Tensor([i for i in range(1, prototypes.shape[0] + 1)]).long().cuda()
                    # import copy
                    # proto_labels = copy.deepcopy(labels)
                    features = torch.Tensor([]).cuda()  # prototypes.repeat(1,2).reshape(prototypes.shape[0],2,-1).cuda()#torch.Tensor([]).cuda()#torch.cat((prototypes.unsqueeze(dim=1), aug_gaussian_fea(prototypes, stds, fea_num=1).reshape(prototypes.shape[0],-1).unsqueeze(dim=1)), dim=1)#torch.Tensor([]).cuda()#prototypes.repeat(1,2).reshape(prototypes.shape[0],2,-1)prototypes.unsqueeze(dim=1)#torch.Tensor([]).cuda()#
                    for ca in cates:
                        if ca<=self.n_classes-self.n_new_class-1:#ca > 0 and
                            ca_fea = pooled_feat[rois_label.eq(ca).nonzero().view(-1)]
                            ca_fea_org = pooled_feat_org[rois_label.eq(ca).nonzero().view(-1)]
                            if mean:
                                cat_fea = torch.cat(
                                    (torch.mean(ca_fea, dim=0).unsqueeze(dim=0), torch.mean(ca_fea_org, dim=0).unsqueeze(dim=0)),
                                    dim=0).unsqueeze(dim=0)
                                # cat_fea = torch.cat((torch.mean(ca_fea,dim=0).unsqueeze(dim=0),torch.mean(ca_fea_org,dim=0).unsqueeze(dim=0)), dim=0).unsqueeze(dim=1)
                            else:
                                cat_fea = torch.cat((ca_fea.unsqueeze(dim=1), ca_fea_org.unsqueeze(dim=1)), dim=1)
                        else:
                            ca_fea = pooled_feat[rois_label.eq(ca).nonzero().view(-1)]
                            if mean:
                                cat_fea = torch.cat((
                                    torch.mean(ca_fea, dim=0).unsqueeze(dim=0), torch.mean(ca_fea, dim=0).unsqueeze(dim=0)),
                                    dim=0).unsqueeze(dim=0)
                                # cat_fea = torch.cat((torch.mean(ca_fea,dim=0).unsqueeze(dim=0),torch.mean(ca_fea_org,dim=0).unsqueeze(dim=0)), dim=0).unsqueeze(dim=1)
                            else:
                                cat_fea = torch.cat((ca_fea.unsqueeze(dim=1), ca_fea.unsqueeze(dim=1)), dim=1)
                        if features.shape[0] > 0:
                            features = torch.cat((features, cat_fea), dim=0)
                            if mean:
                                labels = torch.cat((labels, torch.Tensor([ca]).long().cuda()), dim=0)
                                # labels = torch.cat((labels, torch.Tensor([ca]).long().cuda()), dim=0)
                            else:
                                labels = torch.cat((labels, torch.Tensor([ca for ica in range(0, cat_fea.shape[0])]).long().cuda()), dim=0)
                        else:
                            features = cat_fea
                            if mean:
                                labels = torch.Tensor([ca]).long().cuda()
                            else:
                                labels = torch.Tensor([ca for ica in range(0, cat_fea.shape[0])]).long().cuda()
                else:
                    cates = rois_label[(rois_label > 0).nonzero().view(-1)].unique()
                    # labels = torch.Tensor([i for i in range(1,prototypes.shape[0]+1)]).long().cuda()
                    # import copy
                    # proto_labels = copy.deepcopy(labels)
                    features = torch.Tensor([]).cuda()#prototypes.repeat(1,2).reshape(prototypes.shape[0],2,-1).cuda()#torch.Tensor([]).cuda()#torch.cat((prototypes.unsqueeze(dim=1), aug_gaussian_fea(prototypes, stds, fea_num=1).reshape(prototypes.shape[0],-1).unsqueeze(dim=1)), dim=1)#torch.Tensor([]).cuda()#prototypes.repeat(1,2).reshape(prototypes.shape[0],2,-1)prototypes.unsqueeze(dim=1)#torch.Tensor([]).cuda()#
                    for ca in cates:
                        ca_fea = pooled_feat[rois_label.eq(ca).nonzero().view(-1)]
                        ca_fea_org = pooled_feat_org[rois_label.eq(ca).nonzero().view(-1)]
                        if mean:
                            cat_fea = torch.cat((torch.mean(ca_fea,dim=0).unsqueeze(dim=0),torch.mean(ca_fea_org,dim=0).unsqueeze(dim=0)), dim=0).unsqueeze(dim=0)
                            # cat_fea = torch.cat((torch.mean(ca_fea,dim=0).unsqueeze(dim=0),torch.mean(ca_fea_org,dim=0).unsqueeze(dim=0)), dim=0).unsqueeze(dim=1)
                        else:
                            cat_fea = torch.cat((ca_fea.unsqueeze(dim=1), ca_fea_org.unsqueeze(dim=1)), dim=1)
                        if features.shape[0]>0:
                            features = torch.cat((features, cat_fea), dim=0)
                            if mean:
                                labels = torch.cat((labels, torch.Tensor([ca]).long().cuda()), dim=0)
                                # labels = torch.cat((labels, torch.Tensor([ca]).long().cuda()), dim=0)
                            else:
                                labels = torch.cat((labels, torch.Tensor([ca for ica in range(0, cat_fea.shape[0])]).long().cuda()), dim=0)
                        else:
                            features = cat_fea
                            if mean:
                                labels = torch.Tensor([ca]).long().cuda()
                            else:
                                labels = torch.Tensor([ca for ica in range(0, cat_fea.shape[0])]).long().cuda()
                        # features = torch.cat((features,cat_fea),dim=0)
                        # labels = torch.cat((labels, torch.Tensor([ca]).long().cuda()),dim=0)

            # fg_idx = rois_label.ne(0).nonzero().view(-1)
            # features = pooled_feat[fg_idx].unsqueeze(dim=2)#torch.cat((pooled_feat[fg_idx],pooled_feat_org[fg_idx]),dim=0).unsqueeze(dim=2)
            # # labels: [bsz]
            # labels = rois_label[fg_idx]#torch.cat((rois_label[fg_idx],rois_label[fg_idx]),dim=0)
            # SupContrast
            # print(features.shape)
            if len(features.shape)<2:
                print(features.shape)
            elif len(features.shape)==2:
                features = features.unsqueeze(dim=1)
                features = features.div(torch.norm(features,p=2,dim=2,keepdim=True)) #### ydb normalize
                cts_loss = criterion(features, labels)
            else:
                features = features.div(torch.norm(features, p=2, dim=2, keepdim=True))  #### ydb normalize
                cts_loss = criterion(features, labels)
                # cts_loss = criterion(features)
                if step % 100 == 0:
                    print('cts_loss: %.4f' % (cts_loss), features.shape, labels.shape)
            # # or SimCLR
            # loss = criterion(features)
            # ###########################################################

        pooled_feat = pooled_feat.mean(3).mean(2)
        pooled_feat_org = pooled_feat_org.mean(3).mean(2)

        ############ compute org_rcnn bbox_pred and cls_score ####################

        # ### random select prototypes of old classes #############
        # rand_idx = torch.Tensor([random.randint(0,prototypes.shape[0]-1) for i in range(5)]).cuda().long()
        # pooled_feat = torch.cat((prototypes[rand_idx],pooled_feat),dim=0)
        # rois_label = torch.cat((proto_labels[rand_idx],rois_label),dim=0)
        # #########################################################
        bbox_pred_org_rcnn=fasterRCNN_org.RCNN_bbox_pred(pooled_feat)
        cls_score_org_rcnn=fasterRCNN_org.RCNN_cls_score(pooled_feat)
        #if self.training and not self.class_agnostic:
        ##########################################################################

        # compute bbox offset
        bbox_pred_old = self.RCNN_bbox_pred(pooled_feat)
        # bbox_pred_residual = fasterRCNN_residual.RCNN_bbox_pred(pooled_feat_r)
        ################# split bbox pred (old and new) ###################
        if not self.class_agnostic:
            bbox_pred_new = self.RCNN_bbox_pred_new(pooled_feat)
            bbox_pred_cat = torch.cat((bbox_pred_old, bbox_pred_new), dim=1)
            bbox_pred = bbox_pred_cat
        else:
            bbox_pred = bbox_pred_old
        ###################################################################

        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

            # bbox_pred_view_r = bbox_pred_residual.view(bbox_pred_residual.size(0), int(bbox_pred_residual.size(1) / 4), 4)
            # bbox_pred_select_r = torch.gather(bbox_pred_view_r, 1, rois_label_r.view(rois_label_r.size(0), 1, 1).expand(rois_label_r.size(0), 1, 4))
            # bbox_pred_r = bbox_pred_select_r.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        # cls_score_r = fasterRCNN_residual.RCNN_cls_score(pooled_feat_r)
        # cls_prob_r = F.softmax(cls_score_r,1)#torch.cat((cls_score_r[0],cls_score_r[-self.n_new_class:]),dim=0), 1)
        #
        # cls_score_r_roiinc = fasterRCNN_residual.RCNN_cls_score(pooled_feat_r_roiinc)



        ################# split score fc (old and new)#####################
        cls_score_new = self.RCNN_cls_score_new(pooled_feat)
        cls_score_cat = torch.cat((cls_score, cls_score_new),dim=1)
        cls_prob = F.softmax(cls_score_cat,1)
        ###################################################################





        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:

            if isda_criterion is not None and cfg.TRAIN.pool is False and cfg.TRAIN.feadim==2048:# and idx_fgold.shape[0]>0:
                if not cfg.TRAIN.excls:
                    ######################################################################################################
                    ratio = 0.5 * (epoch / 10)  # 0.5 #* (epoch / (training_configurations[args.model]['epochs']))
                    isda_loss = torch.Tensor([0]).cuda()
                    class_fea = []
                    class_target = []

                    # idx_fgold_gt = ((rois_label_gt > 0) & (rois_label_gt <= self.n_classes - self.n_new_class - 1)).nonzero().view(-1)
                    # idx_fgold_gt = ((rois_label_gt > 0) ).nonzero().view(-1)
                    # if idx_fgold_gt.shape[0]>0:
                    #     gt_labels = rois_label_gt[idx_fgold_gt]
                    #     for bs_idx in range(base_feat_org.shape[0]):
                    #         for roi_idx, roi in enumerate(rois_gt[bs_idx][idx_fgold_gt]):
                    #             roi_fea_org = base_feat[bs_idx][:, roi[2].long()*1/16:roi[4].long()*1/16, roi[1].long()*1/16:roi[3].long()*1/16]
                    #             # roi_fea_org = roi_fea_org.mean(2).mean(1)
                    #             avgpool = nn.AdaptiveAvgPool2d(1)#
                    #             try:
                    #                 roi_fea_org = self.extra_fea(roi_fea_org.unsqueeze(dim=0))
                    #                 roi_fea_org = avgpool(roi_fea_org.squeeze(dim=0)).view(-1)
                    #             except Exception as e:
                    #                 print(e, 'feashape', roi_fea_org.shape)
                    #                 continue
                    #             class_fea.append(roi_fea_org)  # torch.cat((class_fea_old, class_fea_new),dim=0)
                    #             class_target.append(gt_labels[roi_idx])
                    #     if len(class_fea)>0:
                    #         class_fea = torch.stack(class_fea)
                    #         class_target = torch.stack(class_target)
                    #         isda_loss, isda_output = isda_criterion(class_fea, self.extra_classifier, class_fea,
                    #                                             class_target - 1, ratio)
                    # else:
                    #     print('idx_fgold==0')

                    all = True#False#
                    use_gt = False#True
                    bg = True
                    if all:
                        if use_gt:
                            idx_roi = (rois_label_gt > 0).nonzero().view(-1)
                        else:
                            idx_roi = (rois_label > 0).nonzero().view(-1)
                            idx_bg = (rois_label == 0).nonzero().view(-1)
                    else:
                        if use_gt:
                            idx_roi = ((rois_label_gt > 0) & (rois_label_gt <= self.n_classes - self.n_new_class - 1)).nonzero().view(-1)
                        else:
                            idx_roi = ((rois_label > 0) & (rois_label <= self.n_classes - self.n_new_class - 1)).nonzero().view(-1)
                            idx_bg = (rois_label == 0).nonzero().view(-1)
                    if use_gt:
                        pooled_feat_gt = self.RCNN_roi_pool(base_feat, rois_gt.view(-1, 5))
                        pooled_feat_org_gt = fasterRCNN_org.RCNN_roi_pool(base_feat_org, rois_gt.view(-1, 5))
                        pooled_feat_gt = self._head_to_tail(pooled_feat_gt)
                        pooled_feat_org_gt = fasterRCNN_org._head_to_tail(pooled_feat_org_gt)
                        pooled_feat_gt = pooled_feat_gt.mean(3).mean(2)
                        pooled_feat_org_gt = pooled_feat_org_gt.mean(3).mean(2)
                        if idx_roi.shape[0]>0:
                            class_fea_old = pooled_feat_org_gt[idx_roi]
                            class_target = rois_label_gt[idx_roi]
                            class_fea_new = pooled_feat_gt[idx_roi]
                            class_fea = class_fea_old#torch.cat((class_fea_old, class_fea_new),dim=0)
                            class_target = class_target#torch.cat((class_target, class_target),dim=0)
                    else:
                        if bg and idx_roi.shape[0]>0:
                            class_fea_old = pooled_feat_org
                            class_fea_new = pooled_feat
                            class_fea = torch.cat((class_fea_new[idx_roi], class_fea_new[idx_bg]),dim=0)  # torch.cat((class_fea_old, class_fea_new),dim=0)
                            class_target = torch.cat((rois_label[idx_roi], rois_label[idx_bg]),dim=0)  #rois_label#[idx_fgold]
                            # # class_target = torch.cat((class_target, class_target),dim=0)
                        elif idx_roi.shape[0]>0:
                            class_fea_old = pooled_feat_org[idx_roi]
                            class_fea_new = pooled_feat[idx_roi]
                            class_fea = class_fea_old  #torch.cat((class_fea_old, class_fea_new),dim=0)
                            class_target = rois_label[idx_roi]
                            # # class_target = torch.cat((class_target, class_target),dim=0)
                    if len(class_fea)>0 and len(class_target)>0:
                        kg = isda_tmp[0]
                        out_new = isda_tmp[1]
                        feature_mean = isda_tmp[2]

                        if bg:
                            isda_loss, isda_output = isda_criterion(class_fea, (self.RCNN_cls_score, self.RCNN_cls_score_new), class_fea,
                                                                    class_target, ratio, bg=True, kg=kg, out_new=out_new,
                                                                    feature_mean=feature_mean)
                            # isda_loss, isda_output = isda_criterion(class_fea, (self.RCNN_cls_score, self.RCNN_cls_score_new), class_fea,
                            #                                 class_target, ratio, bg=True)
                        else:
                            isda_loss, isda_output = isda_criterion(class_fea, self.RCNN_cls_score, class_fea,
                                                                    class_target - 1, ratio, kg=kg, out_new=out_new,
                                                                    feature_mean=feature_mean)

                            # isda_loss, isda_output = isda_criterion(class_fea, self.RCNN_cls_score, class_fea, class_target-1, ratio)
                        # isda_loss, isda_output = isda_criterion(class_fea, self.RCNN_cls_score,
                        #                                         class_fea,
                        #                                         class_target, ratio)
                    #####################################################################################################
                else:
                    ratio = 0.5 * (epoch / 10)  # 0.5 #* (epoch / (training_configurations[args.model]['epochs']))
                    class_fea = []
                    class_target = []
                    all = True
                    use_gt = True  # False#
                    if all:
                        if use_gt:
                            idx_roi = ((rois_label_gt > 0)).nonzero().view(-1)
                        else:
                            idx_roi = ((rois_label > 0)).nonzero().view(-1)
                    else:
                        if use_gt:
                            idx_roi = ((rois_label_gt > 0) & (
                                        rois_label_gt <= self.n_classes - self.n_new_class - 1)).nonzero().view(-1)
                        else:
                            idx_roi = ((rois_label > 0) & (
                                        rois_label <= self.n_classes - self.n_new_class - 1)).nonzero().view(-1)
                    if use_gt:
                        pooled_feat_gt = self.RCNN_roi_pool(base_feat, rois_gt.view(-1, 5))
                        pooled_feat_org_gt = fasterRCNN_org.RCNN_roi_pool(base_feat_org, rois_gt.view(-1, 5))
                        pooled_feat_gt = self._head_to_tail(pooled_feat_gt)
                        pooled_feat_org_gt = fasterRCNN_org._head_to_tail(pooled_feat_org_gt)
                        pooled_feat_gt_ex = pooled_feat_gt  # self.extra_fea_pool(pooled_feat_gt)
                        pooled_feat_org_gt_ex = pooled_feat_org_gt  # self.extra_fea_pool(pooled_feat_org_gt)
                        pooled_feat_gt_ex = pooled_feat_gt_ex.mean(3).mean(2)
                        pooled_feat_org_gt_ex = pooled_feat_org_gt_ex.mean(3).mean(2)
                        # pooled_feat_gt = self.extra_fea_pool(pooled_feat_gt)
                        # pooled_feat_org_gt = self.extra_fea_pool(pooled_feat_org_gt)
                        if idx_roi.shape[0] > 0:
                            class_fea_old = pooled_feat_org_gt_ex[idx_roi]
                            class_target = rois_label_gt[idx_roi]
                            class_fea_new = pooled_feat_gt_ex[idx_roi]
                            if epoch < 0:
                                class_fea = class_fea_old  # torch.cat((class_fea_old, class_fea_new),dim=0)#class_fea_old#
                                class_target = class_target  # torch.cat((class_target, class_target),dim=0)#class_target#
                            else:
                                class_fea = class_fea_new  # torch.cat((class_fea_old, class_fea_new),dim=0)#class_fea_old#
                                class_target = class_target  # torch.cat((class_target, class_target),dim=0)#class_target#
                    else:
                        if idx_roi.shape[0] > 0:
                            class_fea_old = pooled_feat_org[idx_roi]
                            class_fea_new = pooled_feat[idx_roi]
                            class_target = rois_label[idx_roi]
                            class_fea = class_fea_old  # torch.cat((class_fea_old, class_fea_new),dim=0)
                            # # class_target = torch.cat((class_target, class_target),dim=0)
                            # class_fea = self.extra_fea_pool(class_fea)
                            class_fea = class_fea.mean(3).mean(2)

                    if len(class_fea) > 0 and len(class_target) > 0:
                        # if epoch > 6:
                        #     self.extra_classifier.requires_grad = False
                        # isda_loss, isda_output = isda_criterion(class_fea, (self.RCNN_cls_score, self.RCNN_cls_score_new), class_fea,
                        #                                     class_target, ratio)
                        extra_logit = self.extra_classifier(class_fea_old)
                        extra_logit_new = self.extra_classifier(class_fea_new)

                        ce_loss = F.cross_entropy(extra_logit, class_target - 1)
                        # isda_loss, isda_output = isda_criterion(class_fea, self.extra_classifier, class_fea, class_target-1, ratio)

                        kg = isda_tmp[0]
                        out_new = isda_tmp[1]
                        feature_mean = isda_tmp[2]
                        isda_loss, isda_output = isda_criterion(class_fea, self.extra_classifier, class_fea,
                                                                class_target - 1, ratio, kg=kg, out_new=out_new,
                                                                feature_mean=feature_mean)

                        # cv_var = isda_criterion.get_cv().cuda()
                        # cv_var.requried_grad=True
                        # isda_criterion(self.extra_classifier, class_fea, torch.argmax(extra_logit, dim=1)[0], class_target-1, alpha, weights, cv_var, "update", kg, out_new,
                        #  feature_mean, beta, args.head)

                        # isda_loss += ce_loss
                        # isda_loss, isda_output = isda_criterion(class_fea, self.RCNN_cls_score,
                        #                                         class_fea,
                        #                                         class_target, ratio)
                        # pooled_feat_gt = self._head_to_tail(pooled_feat_gt)
                        # pooled_feat_org_gt = fasterRCNN_org._head_to_tail(pooled_feat_org_gt)
                        pooled_feat_gt = pooled_feat_gt.mean(3).mean(2)
                        pooled_feat_org_gt = pooled_feat_org_gt.mean(3).mean(2)
                        rcnn_logit = self.RCNN_cls_score(pooled_feat_gt[idx_roi])
                        # rcnn_logit_new = self.RCNN_cls_score_new(pooled_feat_gt[idx_roi])
                        # rcnn_logit = torch.cat((rcnn_logit, rcnn_logit_new),dim=1)
                        rcnn_logit_org = fasterRCNN_org.RCNN_cls_score(pooled_feat_org_gt[idx_roi])

                        logp_x = F.log_softmax(extra_logit[:, :rcnn_logit_org[:, 1:].shape[1]], 1)
                        p_y = F.softmax(rcnn_logit_org[:, 1:], 1)
                        logp_x = F.log_softmax(rcnn_logit[:, 1:], 1)
                        p_y = F.softmax(extra_logit[:, :rcnn_logit_org[:, 1:].shape[1]], 1)
                        old_new_logit_loss = F.l1_loss(extra_logit_new[:, :rcnn_logit_org[:, 1:].shape[1]],
                                                       extra_logit[:, :rcnn_logit_org[:, 1:].shape[1]])
                        logit_loss = F.kl_div(logp_x,
                                              p_y)  # F.l1_loss(extra_logit, rcnn_logit_org[:,1:]) #F.l1_loss(rcnn_logit[:,1:], extra_logit)+
                        isda_loss += (logit_loss + old_new_logit_loss)
                        if step % 100 == 0:
                            print('ce_loss: %.4f, old_new_logit_loss: %.4f, logit_loss: %.4f' % (ce_loss, old_new_logit_loss, logit_loss))

            '''
            ############ split ce #############
            max_c=torch.max(cls_score,dim=1)[0]
            cls_score_cat_new = torch.cat((max_c.view(-1,1), cls_score_new), dim=1)
            rois_label_new=rois_label.eq(20).long()#.nonzero()
            #roi_keep_s = rois_label[roi_keep_new.view(-1).long()]
            #cls_score_cat_s = cls_score_cat_s[roi_keep_new.view(-1).long()]
            RCNN_loss_cls_new = F.cross_entropy(cls_score_cat_new, rois_label_new)
            #RCNN_loss_cls_old = F.cross_entropy(cls_score, rois_label)
            ###################################
            '''

            #alpha = 1/20
            # classification loss
            #RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            RCNN_loss_cls = F.cross_entropy(cls_score_cat, rois_label)################## split old and new
            # RCNN_loss_cls = torch.Tensor([0]).cuda()#isda_loss
            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred[-rois_target.shape[0]:], rois_target, rois_inside_ws, rois_outside_ws)

            # RCNN_loss_cls_r = F.cross_entropy(cls_score_r, rois_label_r)  ################## split old and new
            # # bounding box regression L1 loss
            # RCNN_loss_bbox_r = _smooth_l1_loss(bbox_pred_r, rois_target_r, rois_inside_ws_r, rois_outside_ws_r)

            #rcnn_cls_distil_loss=0
            ################### distillation loss #################
            l1_loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)###L1Loss
            cls_score_remove_add_cls=cls_score#[:,:-1]  # split old and new

            #l1_loss = torch.nn.L1Loss(reduce=True, size_average=True)
            #rcnn_cls_distil_loss=l1_loss_fn(cls_score_remove_add_cls,cls_score_org_rcnn) ### L2 loss
            #rcnn_cls_distil_loss=l1_loss_fn(F.softmax(cls_score_remove_add_cls,1),F.softmax(cls_score_org_rcnn,1))
            #rcnn_cls_distil_loss+=l1_loss_fn(F.softmax(cls_score_new,1),F.softmax(cls_score_r[:,1:],1))
            #rcnn_cls_distil_loss = l1_loss(cls_score_remove_add_cls, cls_score_org_rcnn)

            ##### ce loss
            #rcnn_cls_distil_loss = l1_loss_fn(cls_score_remove_add_cls[:,1:], cls_score_org_rcnn[:,1:])


            old_cls_score = cls_score#[:,1:] #cls_score[:,1:]#
            old_cls_score_org = cls_score_org_rcnn#[:,1:]#cls_score_org_rcnn[:,1:]#
            old_cls_score_softmax = F.softmax(old_cls_score, 1)
            old_cls_score_org_softmax = F.softmax(old_cls_score_org, 1)
            rcnn_cls_distil_loss = cfg.TRAIN.cdsllam*l1_loss_fn(old_cls_score_softmax, old_cls_score_org_softmax)
            # if idx_fgold.shape[0]>0:
            #     rcnn_cls_distil_loss = l1_loss_fn(F.softmax(old_cls_score[idx_fgold], dim=1), F.softmax(old_cls_score_org[idx_fgold], dim=1))
            # else:
            #     rcnn_cls_distil_loss = torch.Tensor([0]).cuda()
            # rcnn_cls_distil_loss = l1_loss_fn(old_cls_score, old_cls_score_org)
            if exsup and isda_criterion and extra_logit_exsup is not None:
                idx_fg = ((rois_label > 0)).nonzero().view(-1)
                out_loss = F.kl_div(F.log_softmax(cls_score_cat[:,1:][idx_fg], dim=1), F.softmax(extra_logit_exsup[idx_fg], dim=1))
                if cfg.TRAIN.rdc:
                    fea_loss = F.l1_loss(F.avg_pool1d(pooled_feat.unsqueeze(dim=1), kernel_size=4,
                                                         stride=4).squeeze(dim=1)[idx_fg],
                                           pooled_feat_org_exsup[idx_fg])
                else:
                    fea_loss = F.l1_loss(pooled_feat[idx_fg], pooled_feat_org_exsup[idx_fg])

                if step%100==0:
                    print('dslexsup: output: %.4f, fea: %.4f, isda: %.4f' % (out_loss, fea_loss, isda_loss))
                isda_loss += (out_loss + fea_loss)

                ################### bbox distillation loss ############
            # bbox_pred_residual_roiinc = fasterRCNN_residual.RCNN_bbox_pred(pooled_feat_r_roiinc)
            if ((rois_label>0) & (rois_label<self.n_classes-self.n_new_class)).nonzero().view(-1).shape[0]>0:
                rcnn_bbox_distil_loss = F.smooth_l1_loss(bbox_pred_old[((rois_label>0) & (rois_label<self.n_classes-self.n_new_class)).nonzero().view(-1)],bbox_pred_org_rcnn[((rois_label>0) & (rois_label<self.n_classes-self.n_new_class)).nonzero().view(-1)])### l1 loss[:,4:]
            else:
                rcnn_bbox_distil_loss = torch.Tensor([0]).cuda()
                #rcnn_bbox_distil_loss = l1_loss(bbox_pred_old, bbox_pred_org_rcnn)
            # rcnn_bbox_distil_loss+=l1_loss_fn(bbox_pred_new,bbox_pred_residual_roiinc[:,4:])
            #######################################################

        cls_prob = cls_prob[-rois_target.shape[0]:]
        bbox_pred = bbox_pred[-rois_target.shape[0]:]

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        # cls_prob_r = cls_prob_r.view(batch_size, rois_r.size(1), -1)
        # bbox_pred_r = bbox_pred_r.view(batch_size, rois_r.size(1), -1)
        proto_loss = torch.Tensor([0]).cuda()
        idx_proto = idx_fgold#(rois_label>0).nonzero().view(-1)
        if cfg.TRAIN.proto and prototypes is not None and idx_proto.shape[0]>0:
            '''
            hingeloss = nn.HingeEmbeddingLoss(1)#2)
            # distance = torch.cdist(F.normalize(pooled_feat[idx_proto],dim=-1), F.normalize(prototypes,dim=-1))#, 10)
            distance = 1 - torch.mm(F.normalize(pooled_feat[idx_proto],dim=-1), F.normalize(prototypes,dim=-1).t())#, 10)
            h_labels = []
            for idxfg in idx_proto:#idx_fgold:
                for iproto, proto in enumerate(prototypes):
                    if rois_label[idxfg]==(iproto+1):
                        h_labels.append(1)
                    else:
                        h_labels.append(-1)
            proto_loss = hingeloss(distance, torch.Tensor(h_labels).view(-1, prototypes.shape[0]).cuda())
            # proto_loss = F.l1_loss(pooled_feat[idx_fgold], prototypes[rois_label[idx_fgold]-1])
            '''
            pooled_feat_x = F.normalize(pooled_feat[idx_proto], dim=-1, p=2)
            proto_y = F.normalize(prototypes[rois_label[idx_proto]-1], dim=-1, p=2)
            proto_loss = (2 - 2 * (pooled_feat_x * proto_y).sum(dim=-1)).mean()

        ##### ablation ##############
        # rcnn_cls_distil_loss=torch.Tensor([0]).cuda()
        #rcnn_cls_distil_loss=RCNN_loss_cls_s
        rcnn_bbox_distil_loss=torch.Tensor([0]).cuda()
        base_feat_distil_loss=torch.Tensor([0]).cuda()
        #RCNN_loss_cls=RCNN_loss_cls+RCNN_loss_cls_new
        att_loss = torch.Tensor([0]).cuda()
        rpn_conv1_distil_loss = att_loss #+ pool_base_feat_distil_loss
        # rpn_conv1_distil_loss = torch.Tensor([0]).cuda()
        #pooled_feat_distil_loss+=pooled_feat_distil_loss_1
        pooled_feat_distil_loss = cfg.TRAIN.pdsllam*l1_pooled_fgold_loss#torch.Tensor([0]).cuda()
        base_feat_residual_loss = torch.Tensor([0]).cuda()
        # correlation_loss=torch.Tensor([0]).cuda()
        # correlation_loss = l1_pooled_loss
        if not cfg.TRAIN.CTS:
            cts_loss = torch.Tensor([0]).cuda()
        correlation_loss = cfg.TRAIN.LAMBDA*cts_loss # cts_loss#torch.Tensor([0]).cuda()#
        # pooled_feat_distil_loss = loss_pt #+ l1_pooled_fgold_loss
        # if idx_fgold.shape[0]==0:
        #     l1_pooled_fgold_loss = torch.Tensor([0]).cuda()
        # else:
        #     l1_pooled_fgold_loss = l1_loss_fn(pooled_feat[idx_fgold], pooled_feat_org[idx_fgold])
        # pooled_feat_distil_loss = l1_pooled_fgold_loss
        # print(pooled_feat_distil_loss,loss_pt)

        '''
        cates_old = rois_label[((rois_label > 0) & (rois_label<=self.n_classes-self.n_new_class-1)).nonzero().view(-1)].unique()#rois_label[((rois_label > 0) & (rois_label<=self.n_classes-self.n_new_class-1)).nonzero().view(-1)].unique()
        pooled_feat_aug = pooled_feat[((rois_label > 0) & (rois_label<=self.n_classes-self.n_new_class-1)).nonzero().view(-1)].clone()
        rois_label_aug = rois_label[((rois_label > 0) & (rois_label<=self.n_classes-self.n_new_class-1)).nonzero().view(-1)].clone()
        if cates_old.shape[0]>1:
            for ca in cates_old:
                af_tmp = aug_gaussian_fea(torch.zeros(2048).cuda(), stds[ca-1:ca], pooled_feat_aug[rois_label_aug.eq(ca).nonzero().view(-1)].shape[0])
                # print(af_tmp)
                pooled_feat_aug[rois_label_aug.eq(ca).nonzero().view(-1)] +=  af_tmp.reshape(-1,2048)
            cls_score_aug = self.RCNN_cls_score(pooled_feat_aug)
            cls_score_new_aug = self.RCNN_cls_score_new(pooled_feat_aug)
            cls_score_cat_aug = torch.cat((cls_score_aug, cls_score_new_aug), dim=1)
            RCNN_loss_cls_aug = F.cross_entropy(cls_score_cat_aug, rois_label_aug)
            rcnn_bbox_distil_loss_aug = F.smooth_l1_loss(self.RCNN_bbox_pred(pooled_feat_aug), fasterRCNN_org.RCNN_bbox_pred(pooled_feat_aug))
        else:
            RCNN_loss_cls_aug = torch.tensor(0.0).cuda()
            rcnn_bbox_distil_loss_aug = torch.tensor(0.0).cuda()
        '''
        # RCNN_loss_cls += \
        # rcnn_cls_distil_loss = RCNN_loss_cls_aug
        # rcnn_bbox_distil_loss = rcnn_bbox_distil_loss_aug
        # cates_old = rois_label[
        #     ((rois_label > 0) & (rois_label <= self.n_classes - self.n_new_class - 1)).nonzero().view(
        #         -1)].unique()  # rois_label[((rois_label > 0) & (rois_label<=self.n_classes-self.n_new_class-1)).nonzero().view(-1)].unique()
        #
        # if cates_old.shape[0] > 1:
        #     pooled_old_fea = pooled_feat[((rois_label > 0) & (rois_label <= self.n_classes - self.n_new_class - 1)).nonzero().view(-1)]
        #     # prototypes_old = prototypes[rois_label[((rois_label > 0) & (rois_label <= self.n_classes - self.n_new_class - 1)).nonzero().view(-1)]-1]
        #     dist_cos = torch.mm(pooled_old_fea.div(torch.norm(pooled_old_fea,p=2,dim=1,keepdim=True)), \
        #                         prototypes.div(torch.norm(prototypes,p=2,dim=1,keepdim=True)).t())
        #     # pt_score = self.RCNN_cls_score(pooled_old_fea)
        #     # dist_cos = torch.mm(pooled_old_fea, prototypes.t())
        #     RCNN_loss_cls_aug = F.cross_entropy(dist_cos, rois_label[((rois_label > 0) & (rois_label <= self.n_classes - self.n_new_class - 1)).nonzero().view(-1)]-1)
        # else:
        #     RCNN_loss_cls_aug = torch.tensor(0.0).cuda()
        # # base_feat_distil_loss = l1_loss_fn(base_feat, base_feat_org)
        # rcnn_cls_distil_loss = RCNN_loss_cls_aug
        if bg_coords is not None and cfg.TRAIN.mixre:
            proto_loss = mix_relation_loss
        # if cfg.TRAIN.isda:
        #     return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, \
        #            rcnn_cls_distil_loss, rcnn_bbox_distil_loss, base_feat_distil_loss, rpn_conv1_distil_loss, pooled_feat_distil_loss, correlation_loss, att_loss, isda_loss, mixup_loss, proto_loss, extra_logit
        # else:
        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, \
               rcnn_cls_distil_loss, rcnn_bbox_distil_loss, base_feat_distil_loss, rpn_conv1_distil_loss, pooled_feat_distil_loss, correlation_loss, att_loss, isda_loss, mixup_loss, proto_loss#, None

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                if cfg.TRAIN.bias:
                    m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
