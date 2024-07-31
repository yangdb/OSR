import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.config import cfg

torch.manual_seed(cfg.RNG_SEED)
torch.cuda.manual_seed(cfg.RNG_SEED)
torch.cuda.manual_seed_all(cfg.RNG_SEED)
torch.backends.cudnn.deterministic = True

class EstimatorCV():
    def __init__(self, feature_num, class_num):
        super(EstimatorCV, self).__init__()
        self.class_num = class_num
        self.CoVariance = torch.zeros(class_num, feature_num, feature_num).cuda()
        self.Ave = torch.zeros(class_num, feature_num).cuda()
        self.Amount = torch.zeros(class_num).cuda()

    def update_CV(self, features, labels):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        NxCxFeatures = features.view(
            N, 1, A
        ).expand(
            N, C, A
        )
        onehot = torch.zeros(N, C).cuda()
        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - \
                   ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = torch.bmm(
            var_temp.permute(1, 2, 0),
            var_temp.permute(1, 0, 2)
        ).div(Amount_CxA.view(C, A, 1).expand(C, A, A))

        sum_weight_CV = onehot.sum(0).view(C, 1, 1).expand(C, A, A)

        sum_weight_AV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(
            sum_weight_CV + self.Amount.view(C, 1, 1).expand(C, A, A)
        )
        weight_CV[weight_CV != weight_CV] = 0

        weight_AV = sum_weight_AV.div(
            sum_weight_AV + self.Amount.view(C, 1).expand(C, A)
        )
        weight_AV[weight_AV != weight_AV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul(
            torch.bmm(
                (self.Ave - ave_CxA).view(C, A, 1),
                (self.Ave - ave_CxA).view(C, 1, A)
            )
        )

        self.CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp.mul(weight_CV)).detach() + additional_CV.detach()

        self.Ave = (self.Ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach()

        self.Amount += onehot.sum(0)


class ISDALoss(nn.Module):
    def __init__(self, feature_num, class_num, old_num=0):
        super(ISDALoss, self).__init__()

        self.estimator = EstimatorCV(feature_num, class_num)

        self.class_num = class_num
        self.fea_dim = feature_num
        self.cross_entropy = nn.CrossEntropyLoss()
        self.old_num = old_num

    def isda_aug(self, fc, fc_t, features, y, y_t, labels, cv_matrix, ratio, kg=None, out_new=None, feature_mean=None):

        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        if isinstance(fc, tuple):
            weight_m_1 = list(fc[0].parameters())[0]
            weight_m_2 = list(fc[1].parameters())[0]
            weight_m = torch.cat((weight_m_1, weight_m_2), dim=0)[1:,:]
        else:
            if cfg.TRAIN.excls:
                weight_m = list(fc.parameters())[0]
            else:
                weight_m = list(fc.parameters())[0][1:,:]
        if isinstance(fc_t, tuple):
            weight_m_1_t = list(fc_t[0].parameters())[0]
            weight_m_2_t = list(fc_t[1].parameters())[0]
            weight_m_t = torch.cat((weight_m_1_t, weight_m_2_t), dim=0)[1:,:]
        else:
            if cfg.TRAIN.excls:
                weight_m_t = list(fc_t.parameters())[0]
            else:
                weight_m_t = list(fc_t.parameters())[0][1:,:]
        if weight_m_t.shape[1]==2048 and cfg.TRAIN.rdc:
            weight_m_t = F.avg_pool1d(weight_m_t.unsqueeze(dim=1), kernel_size=4, stride=4).squeeze(dim=1)


        NxW_ij = weight_m.expand(N, C, A)
        NxW_ij_t = weight_m_t.expand(N, C, A)

        NxW_kj = torch.gather(NxW_ij, 1,
                              labels.view(N, 1, 1)
                              .expand(N, C, A))
        NxW_kj_t = torch.gather(NxW_ij_t, 1,
                              labels.view(N, 1, 1)
                              .expand(N, C, A))

        CV_temp = cv_matrix[labels]

        # sigma2 = ratio * \
        #          torch.bmm(torch.bmm(NxW_ij - NxW_kj,
        #                              CV_temp).view(N * C, 1, A),
        #                    (NxW_ij - NxW_kj).view(N * C, A, 1)).view(N, C)

        sigma2 = ratio * \
                 torch.bmm(torch.bmm(NxW_ij - NxW_kj,
                                     CV_temp),
                           (NxW_ij - NxW_kj).permute(0, 2, 1))

        sigma2_t = ratio * \
                 torch.bmm(torch.bmm(NxW_ij_t - NxW_kj_t,
                                     CV_temp),
                           (NxW_ij_t - NxW_kj_t).permute(0, 2, 1))

        sigma2 = sigma2.mul(torch.eye(C).cuda()
                            .expand(N, C, C)).sum(2).view(N, C)

        sigma2_t = sigma2_t.mul(torch.eye(C).cuda()
                            .expand(N, C, C)).sum(2).view(N, C)

        aug_result = y + 0.5 * sigma2
        aug_result_t = y_t + 0.5 * sigma2_t
        out_new = None
        if out_new is not None:
            # reasnoning mu in loss function
            dataMean_NxA = out_new[labels]
            dataMean_NxAx1 = dataMean_NxA.expand(1, N, A).permute(1, 2, 0).cuda()
            del CV_temp
            dataW_NxCxA = NxW_ij - NxW_kj
            dataW_NxCxA_t = NxW_ij_t - NxW_kj_t
            dataW_x_detaMean_NxCx1 = torch.bmm(dataW_NxCxA, dataMean_NxAx1)
            dataW_x_detaMean_NxCx1_t = torch.bmm(dataW_NxCxA_t, dataMean_NxAx1)

            datW_x_detaMean_NxC = dataW_x_detaMean_NxCx1.view(N, C)
            datW_x_detaMean_NxC_t = dataW_x_detaMean_NxCx1_t.view(N, C)

            alpha = 1
            aug_result += alpha * datW_x_detaMean_NxC
            aug_result_t += alpha * datW_x_detaMean_NxC_t

            # print('augresult', aug_result)
        return aug_result, aug_result_t

    # def forward(self, model, fc, x, target_x, ratio):
    #
    #     features = model(x)
    #
    #     y = fc(features)
    #
    #     self.estimator.update_CV(features.detach(), target_x)
    #
    #     isda_aug_y = self.isda_aug(fc, features, y, target_x, self.estimator.CoVariance.detach(), ratio)
    #
    #     loss = self.cross_entropy(isda_aug_y, target_x)
    #
    #     return loss, y
    def forward(self, features, fc, fc_t, x, target_x, ratio, bg=False, kg=None, out_new=None, feature_mean=None):

        # features = model(x)
        if isinstance(fc, tuple):
            y1 = fc[0](features)
            y2 = fc[1](features)
            y = torch.cat((y1, y2), dim=1)[:,1:]
        else:
            if cfg.TRAIN.excls:
                y = fc(features)
            else:
                y = fc(features)[:,1:]

        y_t = y
        # if isinstance(fc_t, tuple):
        #     y1_t = fc_t[0](features) ### 512 vs. 2048
        #     y2_t = fc_t[1](features)
        #     y_t = torch.cat((y1_t, y2_t), dim=1)[:,1:]
        # else:
        #     if cfg.TRAIN.excls:
        #         y_t = fc_t(features)
        #     else:
        #         y_t = fc_t(features)[:, 1:]

        self.estimator.update_CV(features.detach(), target_x)

        if kg is not None:
            cv_var = self.get_cv()
            size1 = cv_var.size(1)
            cv_matrix_temp = cv_var.view(cv_var.size(0), -1).cuda()
            cv_var_new = (cv_var[:self.old_num]+torch.matmul(kg[self.old_num:, :self.old_num].float().t(), cv_matrix_temp[self.old_num:].float()).view(self.old_num, size1, -1))/2
            cv_var = cv_var.cuda()
            cv_var_new = cv_var_new.cuda()
            if self.class_num>self.old_num:
                new_cv = torch.cat((cv_var_new, cv_var[self.old_num:]), 0)
            else:
                new_cv = cv_var_new
                out_new = out_new[:self.old_num]
                feature_mean = feature_mean[:self.old_num]
            cv = new_cv
            self.estimator.CoVariance = cv

        if torch.sum(self.estimator.Amount)>200:
            # try:
            isda_aug_y, isda_aug_y_t = self.isda_aug(fc, fc_t, features, y, y_t, target_x, self.estimator.CoVariance.detach(), ratio, kg=kg, out_new=out_new, feature_mean=feature_mean)
            loss = self.cross_entropy(isda_aug_y, target_x)
            loss += F.kl_div(F.log_softmax(isda_aug_y, 1), F.softmax(isda_aug_y_t, 1))
            # except Exception as e:
            #     print(e,'isdaexception', features.shape)
            #     loss = torch.Tensor([0]).cuda()
        else:
            loss = torch.Tensor([0]).cuda()
            y = None
        return loss, y

    def get_cv(self):
        return self.estimator.CoVariance

    def update_cv(self, cv):
        self.estimator.CoVariance = cv

import numpy as np
class RISDA_CE(nn.Module):
    def __init__(self, feature_num, class_num, cls_num_list, max_m=0.5, s=30):
        super(RISDA_CE, self).__init__()
        self.estimator = EstimatorCV(feature_num, class_num)
        self.class_num = class_num
        self.cross_entropy = nn.CrossEntropyLoss()

        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s

    def RISDA(self, fc_kg_new, features, y_s, labels_s, s_cv_matrix, alpha, kg, out_new, feature_mean, beta):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        weight_m = fc_kg_new
        NxW_ij = weight_m.expand(N, C, A)
        NxW_kj = torch.gather(NxW_ij, 1, labels_s.view(N, 1, 1).expand(N, C, A))

        s_CV_temp = s_cv_matrix[labels_s]
        # use beta calculate sigma_ij
        sigma2 = beta * torch.bmm(torch.bmm(NxW_ij - NxW_kj, s_CV_temp), (NxW_ij - NxW_kj).permute(0, 2, 1))
        sigma2 = sigma2.mul(torch.eye(C).cuda().expand(N, C, C)).sum(2).view(N, C)

        # reasnoning mu in loss function
        dataMean_NxA = out_new[labels_s]

        dataMean_NxAx1 = dataMean_NxA.expand(1, N, A).permute(1, 2, 0).cuda()
        del s_CV_temp
        dataW_NxCxA = NxW_ij - NxW_kj
        dataW_x_detaMean_NxCx1 = torch.bmm(dataW_NxCxA, dataMean_NxAx1)
        datW_x_detaMean_NxC = dataW_x_detaMean_NxCx1.view(N, C)

        aug_result = y_s + 0.5 * sigma2 + alpha * datW_x_detaMean_NxC
        return aug_result

    def forward(self, fc, features, y_s, labels, alpha, weights, cv, manner, kg, out_new, feature_mean, beta, head):
        self.estimator.update_CV(features.detach(), labels)
        # reasoning covariance  head=20
        tail = self.class_num - head
        cv_var = self.get_cv()
        cv_matrix_temp = cv_var.view(cv_var.size(0), -1).cuda()
        kg = kg.cuda()

        cv_var_new = torch.matmul(kg[head:], cv_matrix_temp).view(tail, 64, -1)
        cv_var = cv_var.cuda()
        cv_var_new = cv_var_new.cuda()
        new_cv = torch.cat((cv_var[:head], cv_var_new), 0)
        cv = new_cv
        # update covariance
        self.estimator.CoVariance = new_cv

        fc_kg = list(fc.named_leaves())[0][1]
        fc_kg_new = fc_kg

        aug_y = self.RISDA(fc_kg_new, features, y_s, labels, cv, alpha, kg, out_new, feature_mean, beta)
        loss = F.cross_entropy(aug_y, labels, weight=weights)
        return loss

    def get_cv(self):
        return self.estimator.CoVariance

    def update_cv(self, cv):
        self.estimator.CoVariance = cv