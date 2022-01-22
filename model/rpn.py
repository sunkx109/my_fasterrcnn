import numpy as np
import torch
from torch.nn import functional as F
import torch as t
from torch import nn
from model.utils.bbox_tools import generate_anchor_base
from model.utils.creator_tool import ProposalCreator


class RegionProposalNetwork(nn.Module):
    def __init__(
            self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32], feat_stride=16,
            proposal_creator_params=dict(),
    ):
        super(RegionProposalNetwork, self).__init__()
        # anchor_base.shape = (9,4)
        self.anchor_base = generate_anchor_base(
            anchor_scales=anchor_scales, ratios=ratios)
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        n_anchor = self.anchor_base.shape[0] #9
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1) # rpn网络的第一个3*3 卷积
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0) # 分类部分 (N,512,H,W) -> (N,2*9,H,W)
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)  #回归部分   (N,512,H,W) -> (N,4*9,H,W)
        normal_init(self.conv1, 0, 0.01) 
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        n, _, hh, ww = x.shape # 得到特征图的N,C,H,W
        anchor = _enumerate_shifted_anchor(
            np.array(self.anchor_base),
            self.feat_stride, hh, ww)
        #anchor.shape = (height*width*9,4) 这里的height和width是特征图的height和width
        n_anchor = anchor.shape[0] // (hh * ww) # 9
        h = F.relu(self.conv1(x)) # rpn网络的第一个3*3 卷积

        rpn_locs = self.loc(h) #回归 (N,512,H,W) -> (N,4*9,H,W)
        # UNNOTE: check whether need contiguous
        # A: Yes
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4) #(N,4*9,H,W) -> (N,9*H*W,4)
        
        rpn_scores = self.score(h) #分类 (N,512,H,W) -> (N,2*9,H,W)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous() # (N,2*9,H,W)-> (N,H,W,2*9)
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4) # (N,H,W,2*9)->(N,H,W,9,2) 并在dim=4基础上去计算softmax
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous() # (N,H,W,9)
        rpn_fg_scores = rpn_fg_scores.view(n, -1) # (N,9*H*W)
        rpn_scores = rpn_scores.view(n, -1, 2) # (N,H,W,2*9) ->(N,9*H*W,2)

        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size,
                scale=scale)
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = np.concatenate(rois, axis=0) 
        roi_indices = np.concatenate(roi_indices, axis=0)
        # 得到五个数据 rpn 的回归数据rpn_locs,(N, H*W*A, 4)
        # rpn的分类数据 rpn_scores  (N, H*W*A, 2)
        # N个矩阵的roi数据 组成的rois (N*R,4)
        # roi_indices 对应着roi的编号以及长度
        # anchor (H*W*A,4)
        return rpn_locs, rpn_scores, rois, roi_indices, anchor


def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    # !TODO: add support for torch.CudaTensor
    # xp = cuda.get_array_module(anchor_base)
    # it seems that it can't be boosed using GPU
    
    # 以0为起点 ，height * feat_stride为终点   feat_stride为步长 得到一个np的array
    # 如array [0,16,32]
    # 这里对height和width采用了相同的步长，在torchvision中是区别对待了的
    shift_y = np.arange(0, height * feat_stride, feat_stride)  
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # 这里的代码与torchvision的类似，torch是用其内置方法，这里使用的np的方法
    # 其中np.ravel()方法的效果就是生成一个连续的一维序列，其效果等价与reshape(-1)
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)
    # 得到shift.shape = (height*width,4)
    
    # anchor_base.shape = (9,4)
    A = anchor_base.shape[0] # A=9
    K = shift.shape[0] # K=height*width
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    # anchor.shape = (k*A,4)->(height*width*9,4)
    # 也就是以特征图的每个点为中心的
    return anchor


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
