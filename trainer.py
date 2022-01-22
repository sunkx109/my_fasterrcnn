from __future__ import  absolute_import
import os
from collections import namedtuple
import time
from torch.nn import functional as F
from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator

from torch import nn
import torch
from utils import array_tool as at

from utils.config import opt
from torchnet.meter import ConfusionMeter, AverageValueMeter

LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])


class FasterRCNNTrainer(nn.Module):
    def __init__(self, faster_rcnn):
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = opt.rpn_sigma #3
        self.roi_sigma = opt.roi_sigma #1

        # target creator create gt_bbox gt_label etc as training targets. 
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.optimizer = self.faster_rcnn.get_optimizer()

    def forward(self, imgs, bboxes, labels, scale):
        n = bboxes.shape[0] # N
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        # 得到特征图
        features = self.faster_rcnn.extractor(imgs)
        
        # rpn 返回五个数据
        # rpn 的回归数据rpn_locs (N, features_H*features_W*A, 4)
        # rpn的分类数据 rpn_scores  (N, features_H*features_W*A, 2)
        # roi数据 (N*R,4)
        # roi数据对应的索引(N*R,)
        # 所有的anchor (A*features_H*features_W,4)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.faster_rcnn.rpn(features, img_size, scale)
        """
        这里代码说明一下，主要是batchsize=1 注意这一点
        那么 bboxses.shape = (1,R,4)
             label.shape = (1,R)
             rpn_score.shape = (1,features_H*features_W*A,2)
             roi.shape = (R,4)
             roi_indices.shape = (R)
        """
        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0] # shape = (R',4)
        label = labels[0] # shape = (R',)
        rpn_score = rpn_scores[0] # (features_H*features_W*A,2)
        rpn_loc = rpn_locs[0] # (features_H*features_W*A,4)
        roi = rois #(R,4)

        # Sample RoIs and forward
        # it's fine to break the computation graph of rois, 
        # consider them as constant input
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            at.tonumpy(bbox),
            at.tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)
        # proposal_target_creator做了一个这样的事
        # 通过给定的bbox label 以及roi得到满足阈值的 roi(sample_roi) roc_loc的偏移量(gt_roi_loc) 以及roi的label(gt_roi_label)
        # NOTE it's all zero because now it only support for batch=1 now
        sample_roi_index = torch.zeros(len(sample_roi))
        # head 计算得到最终roi的分类和回归
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features,
            sample_roi,
            sample_roi_index)
        
        # 计算loss
        # ------------------ RPN losses -------------------#
        # gt_rpn_loc 为真值偏移量
        # gt_rpn_label 为真值label
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            at.tonumpy(bbox),
            anchor,
            img_size)
        gt_rpn_label = at.totensor(gt_rpn_label).long()
        gt_rpn_loc = at.totensor(gt_rpn_loc)
        # 第一个loss
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma)

        # NOTE: default value of ignore_index is -100 ...
        # 第二个loss
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]
        # ------------------ ROI losses (fast rcnn loss) -------------------#
        n_sample = roi_cls_loc.shape[0] #128
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4) # [128,21,4]
        roi_loc = roi_cls_loc[torch.arange(0, n_sample).long().cuda(), \
                              at.totensor(gt_roi_label).long()]
        gt_roi_label = at.totensor(gt_roi_label).long()
        gt_roi_loc = at.totensor(gt_roi_loc)
        
        # 第三个loss
        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)
        
        # 第四个loss
        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]

        return LossTuple(*losses)

    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        return losses

# sigma的值rpn中为3
#          roi中为1
def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    # |(x-t)| where in_weight>0 else 0
    abs_diff = diff.abs()
    # 根据smooth L1 的公式
    # if |x|<1 :
    #     smooth L1(x) = 0.5 * x^2
    # else:
    #     smooth L1(x) = |x| - 0.5
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = torch.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation, 
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    # 将gt_label中下标大于0的in_weight设置为1，其余设置为0
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    # 求得平均loss
    loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss
    return loc_loss
