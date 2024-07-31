import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.rpn.bbox_transform import bbox_transform_inv
from model.rpn.bbox_transform import clip_boxes
from model.roi_layers import ROIAlign, ROIPool
from model.roi_layers import nms
# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_align.modules.roi_align import RoIAlignAvg

from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

def compute_iou(box1, box2, iou_thresh=0.3, wh=False):
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
    if iou>iou_thresh:
        return True
    else:
        return False
    #return iou

class _fasterRCNN_inc(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN_inc, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
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

    def forward(self, im_data, im_info, gt_boxes, num_boxes, rois_org,cls_prob_org,bbox_pred_org,rois_label_org,fasterRCNN_org,step,roidb,ratio_index):

        ########## frcnn_org_result #################
        scores = cls_prob_org.data
        boxes = rois_org.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred_org.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if self.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * (self.n_classes-1))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        #pred_boxes /= im_info.data[0][2]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        thresh=0.3#0.05
        #print(scores.shape)
        org_det_gt_boxes=torch.Tensor().cuda()
        for j in range(1, scores.shape[1]):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if self.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                cls_tensor = torch.full([cls_dets.shape[0],1], j).cuda()
                cls_label_cat=torch.cat((cls_dets[:,0:4],cls_tensor),1)
                if org_det_gt_boxes.shape[0]==0:
                    org_det_gt_boxes=cls_label_cat
                else:
                    org_det_gt_boxes=torch.cat((org_det_gt_boxes,cls_label_cat), 0)

        #############################################


        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        ####compute IOU between gt_boxes and rpn_org_proposals, delete overlapped bboxes of rpn_org_proposals
        final_org_det=torch.Tensor().cuda()
        for o_bbox in org_det_gt_boxes:
            uq_flag=True
            for gt_bbox in gt_boxes.squeeze()[:num_boxes]:
                if compute_iou(o_bbox,gt_bbox):
                    uq_flag=False
                    break
            if uq_flag:
                if final_org_det.shape[0]==0:
                    final_org_det=o_bbox.unsqueeze(dim=0)
                else:
                    final_org_det=torch.cat((final_org_det,o_bbox.unsqueeze(dim=0)),0)
        org_det_gt_boxes=final_org_det#.unsqueeze(dim=0)
        ###################################################
        gt_boxes=torch.cat((org_det_gt_boxes,gt_boxes.squeeze()),0).unsqueeze(dim=0)
        num_boxes+=org_det_gt_boxes.shape[0]
        '''
        def tensor_to_PIL(tensor):
            image = tensor.cpu().clone()
            image = image.squeeze(0)
            from torchvision import transforms
            unloader = transforms.ToPILImage()
            image = unloader(image)
            return image
        img=tensor_to_PIL(im_data)
        '''

        ############################## draw gt box ##############################################
        '''
        from PIL import Image, ImageDraw
        #import cv2
        #im=cv2.imread(roidb[ratio_index[step]]['image'])

        #im = cv2.resize(im, (int(im_info[0][1].cpu().item()),int(im_info[0][0].cpu().item())),
        #                interpolation=cv2.INTER_LINEAR)

        # 创建一个可以在给定图像上绘图的对象
        img=Image.open(roidb[ratio_index[step]]['image'])
        img=img.resize((int(im_info[0][1].cpu().item()),int(im_info[0][0].cpu().item())))
        import numpy as np
        if roidb[ratio_index[step]]['flipped']:
            img = Image.fromarray(np.array(img)[:, ::-1, :])
        draw = ImageDraw.Draw(img)

        for idx,g in enumerate(gt_boxes.squeeze()):
            if idx==num_boxes.cpu().item():
                break
            box=g.cpu()
            #cv2.rectangle(im, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (255, 0, 0), -1)
            #cv2.putText(im, self.classes[int(box[4].item())],  (box[0], box[1]),  cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 255), 2)
            draw.rectangle((box[0], box[1], box[2], box[3]), outline=(255, 0, 0))
            draw.text([box[0], box[1]], self.classes[int(box[4].item())], "red")
        #cv2.imwrite('1.jpg',im)
        img.save('1.jpg')
        '''
        ###################################################################################################################
        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)


        ############ compute org_rcnn bbox_pred and cls_score ####################
        #bbox_pred_org_rcnn=fasterRCNN_org.RCNN_bbox_pred(pooled_feat)
        cls_score_org_rcnn=fasterRCNN_org.RCNN_cls_score(pooled_feat)
        #if self.training and not self.class_agnostic:

        ##########################################################################

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        ################# split score fc (old and new)#####################
        cls_score_new = self.RCNN_cls_score_new(pooled_feat)
        cls_score_cat = torch.cat((cls_score,cls_score_new),dim=1)
        cls_prob = F.softmax(cls_score_cat,1)
        ###################################################################


        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            #RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            RCNN_loss_cls = F.cross_entropy(cls_score_cat,rois_label)################## split old and new
            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

            #rcnn_cls_distil_loss=0
            ################### distillation loss #################
            l1_loss_fn = torch.nn.L1Loss(reduce=True, size_average=True)
            cls_score_remove_add_cls=cls_score#[:,:-1]  # split old and new
           # rcnn_cls_distil_loss=l1_loss_fn(cls_score_remove_add_cls,cls_score_org_rcnn) ### L1 loss

            #cls_prob_org_rcnn = F.softmax(cls_score_org_rcnn, 1)
            cls_pred_org_rcnn = cls_score_org_rcnn.argmax(dim=1, keepdim=True).view(-1)
            rcnn_cls_distil_loss = F.cross_entropy(cls_score_remove_add_cls, cls_pred_org_rcnn) ### cross_entropy
            #######################################################


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)



        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label,rcnn_cls_distil_loss

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
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
