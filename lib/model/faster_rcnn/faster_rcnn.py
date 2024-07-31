import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN ### se block
from model.rpn.rpn import _RPN_ori as _RPN

from model.roi_layers import ROIAlign, ROIPool

# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_align.modules.roi_align import RoIAlignAvg

from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

from copy import deepcopy

torch.manual_seed(cfg.RNG_SEED)
np.random.seed(cfg.RNG_SEED)
random.seed(cfg.RNG_SEED)  ##
torch.cuda.manual_seed(cfg.RNG_SEED)
torch.cuda.manual_seed_all(cfg.RNG_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.enabled = False

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
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

    def forward(self, im_data, im_info, gt_boxes, num_boxes, meta_rpn_conv = None, extract_gtfea=False, isda=False, draw=False):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)
        # bfea = self.RCNN_base[4](self.RCNN_base[3](self.RCNN_base[2](self.RCNN_base[1](self.RCNN_base[0](im_data)))))

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes, meta_rpn_conv = meta_rpn_conv)

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
            if extract_gtfea:
                if num_boxes[0] > 0: #gt_boxes.equal(torch.FloatTensor(1).cuda().resize_(1, 1, 5).zero_()) or
                    # print(num_boxes,gt_boxes)
                    gt_boxes_append = gt_boxes[:, :num_boxes].new(gt_boxes[:, :num_boxes].size()).zero_()
                    gt_boxes_append[:, :, 1:5] = gt_boxes[:, :num_boxes, :4]
                    rois = gt_boxes_append  # [:, :num_boxes]
                    rois_label = Variable(gt_boxes[:, :num_boxes, -1].view(-1).long())

                # if gt_boxes.shape[1]>0 and len(num_boxes)>0 and num_boxes[0]>0:
                #     #print(num_boxes)
                #     gt_boxes_append = gt_boxes.new(gt_boxes.size()).zero_()
                #     gt_boxes_append[:, :, 1:5] = gt_boxes[:, :, :4]
                #     rois = gt_boxes_append[:, :num_boxes]
                #     rois_label = Variable(gt_boxes[:,:num_boxes,-1].view(-1).long())
        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if draw:
            ############################## draw gt box ##############################################
            from PIL import Image, ImageDraw
            img = (im_data[0].permute(1,2,0) + torch.from_numpy(cfg.PIXEL_MEANS).cuda().float()).cpu().numpy()
            # print(img)
            img_c = img.copy()
            img_d = Image.fromarray(np.uint8(img_c[:, :, ::-1]))
            a = ImageDraw.ImageDraw(img_d)
            for box in gt_boxes.cpu().numpy()[0]:
                x1, y1, x2, y2 = box[0]+20, box[1]+20, box[2]-20, box[3]-20
                a.rectangle(((x1, y1), (x2, y2)), fill=None, outline='red')
                a.text((x1, y1), str(box[4]), fill=(255, 255, 0))
            import matplotlib.pyplot as plt
            plt.figure("Image")
            plt.imshow(img_d)
            plt.axis('on')
            plt.title('image')
            plt.show()


        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        if not self.training and isda and cfg.TRAIN.feadim==1024:
            return pooled_feat.mean(3).mean(2), rois_label

        if cfg.TRAIN.feadim == 1024 and cfg.TRAIN.isda:
            pooled_feat_1024 = copy.deepcopy(pooled_feat.mean(3).mean(2))

        pooled_feat_copy=None #deepcopy(pooled_feat.detach().reshape(-1))
        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)
        pooled_feat = pooled_feat.mean(3).mean(2)

        if not self.training and isda and cfg.TRAIN.feadim == 2048:
            if cfg.TRAIN.rdc:
                pooled_feat_rdc = F.avg_pool1d(pooled_feat.unsqueeze(dim=1), kernel_size=4, stride=4).squeeze(dim=1)
            cls_score = self.RCNN_cls_score(pooled_feat)
            cls_prob = F.softmax(cls_score, 1)
            return pooled_feat_rdc, rois_label, pooled_feat, cls_prob

        if cfg.TRAIN.feadim == 2048 and cfg.TRAIN.isda:
            pooled_feat_1024 = copy.deepcopy(pooled_feat)
            if cfg.TRAIN.rdc:
                pooled_feat_1024 = F.avg_pool1d(pooled_feat_1024.unsqueeze(dim=1), kernel_size=4, stride=4).squeeze(dim=1)


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

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
        if cfg.TRAIN.isda:
            rois_feat = pooled_feat_1024
        else:
            rois_feat = pooled_feat
        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, rois_feat

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
