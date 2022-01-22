import numpy as np
import torch
from torchvision.ops import nms
from model.utils.bbox_tools import bbox2loc, bbox_iou, loc2bbox


class ProposalTargetCreator(object):
    def __init__(self,
                 n_sample=128,
                 pos_ratio=0.25, pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0
                 ):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo  # NOTE:default 0.1 in py-faster-rcnn

    def __call__(self, roi, bbox, label,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        n_bbox, _ = bbox.shape # (R',4)

        roi = np.concatenate((roi, bbox), axis=0) #shape = (R'+R,4)

        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio) # 128 * 0.5
        # bbox_iou return shape is (N,K)
        # 其中(n,k)位置表示在bbox_a 中的第n个box与在bbox_b 中的第k个box之间的iou
        iou = bbox_iou(roi, bbox) # (R'+R,R')
        gt_assignment = iou.argmax(axis=1) # 每行中最大值所对应的列索引 shape = (R'+R)
        max_iou = iou.max(axis=1) # 每行中最大值所对应的值 shape = (R'+R)
        # Offset range of classes from [0, n_fg_class - 1] to [1, n_fg_class].
        # The label with value 0 is the background.
        gt_roi_label = label[gt_assignment] + 1 

        # Select foreground RoIs as those with >= pos_iou_thresh IoU.
        # 选择那些iou大于阈值0.5的索引
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False)

        # Select background RoIs as those within
        # [neg_iou_thresh_lo, neg_iou_thresh_hi).
        # 对于max_iou在[0,0.5)之间的索引
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) &
                             (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image,
                                         neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False)

        # The indices that we're selecting (both positive and negative).
        keep_index = np.append(pos_index, neg_index) # 将pos_index 和 neg_index组合为一个list
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0  # negative labels --> 0
        sample_roi = roi[keep_index]

        # Compute offsets and scales to match sampled RoIs to the GTs.
        # bbox2loc 是loc2bbox 的逆过程 
        # bbox2loc将roi 与 gt_box 对比计算进而得到dh,dw,dx,dy的真值
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        # 之后对gt_roi_loc做一个归一化
        gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)
                       ) / np.array(loc_normalize_std, np.float32))

        return sample_roi, gt_roi_loc, gt_roi_label


class AnchorTargetCreator(object):
    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):

        img_H, img_W = img_size

        n_anchor = len(anchor)
        # 对anchor做一个简单的裁剪，将超出原始图像HW的anchor裁出
        inside_index = _get_inside_index(anchor, img_H, img_W)
        anchor = anchor[inside_index]
        # argmax_ious为每个anchor与所有bbox计算iou得到最大iou值所对应的bbox的index
        # label 为anchor与bbox经过计算iou后 得到的label
        argmax_ious, label = self._create_label(
            inside_index, anchor, bbox)

        # compute bounding box regression targets
        # 原坐标为anchor 目标坐标为bbox[argmax_ious]
        # 通过bbox2loc这个函数来计算得到偏移量loc 
        # 因为argmax_ious为anchor与bbox计算iou得到的最大iou的索引值列表
        # 所以bbox[argmax_ious]实际上是做了一个anchor与bbox的对应
        # 之后通过bbox2loc计算出来偏移量
        loc = bbox2loc(anchor, bbox[argmax_ious])

        # map up to original set of anchors
        # 将label 和loc 映射会原始anchor对应的index
        label = _unmap(label, n_anchor, inside_index, fill=-1)
        loc = _unmap(loc, n_anchor, inside_index, fill=0)

        return loc, label

    def _create_label(self, inside_index, anchor, bbox):
        # label: 1 is positive, 0 is negative, -1 is dont care
        # inside_index 个anchor 
        label = np.empty((len(inside_index),), dtype=np.int32)
        label.fill(-1)
        
        # 返回值 argmax_ious 为每个anchor与所有bbox计算iou得到最大iou值所对应的bbox的index
        #        max_ious 为每个anchor与所有bbox计算iou得到最大的iou值
        #       gt_argmax_ious 为bbox与anchor求iou的最大值对应的anchor的index
        argmax_ious, max_ious, gt_argmax_ious = \
            self._calc_ious(anchor, bbox, inside_index)

        # assign negative labels first so that positive labels can clobber them
        # 对于max_ious小于neg阈值的anchor的index所对应label的值设置为0
        label[max_ious < self.neg_iou_thresh] = 0

        # positive label: for each gt, anchor with highest iou
        # 对于属于gt_argmax_ious 的anchor的index对应的label则设置为1
        label[gt_argmax_ious] = 1

        # positive label: above threshold IOU
        # 对于max_ious大于pos阈值的anchor的index所对应的label的值设置为1
        label[max_ious >= self.pos_iou_thresh] = 1

        # subsample positive labels if we have too many
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        # subsample negative labels if we have too many
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1
        
        # 返回值说明：
        #    argmax_ious为每个anchor与所有bbox计算iou得到最大iou值所对应的bbox的index
        #    label 为anchor与bbox经过计算iou后 得到的label
        return argmax_ious, label

    def _calc_ious(self, anchor, bbox, inside_index):
        # ious between the anchors and the gt boxes
        # bbox_iou 返回tensor.shape = (n,k),其中n为anchor为
        ious = bbox_iou(anchor, bbox) # shape=(n,k)
        argmax_ious = ious.argmax(axis=1) # 每个anchor与k个bbox的iou最大的下标 shape=(n,)
        max_ious = ious[np.arange(len(inside_index)), argmax_ious] # len(inside_index)个anchor与bbox求得iou的最大值 shape=(n,)
        gt_argmax_ious = ious.argmax(axis=0) # 每个bbox与n个anchor的iou最大的下标 shape=(k,)
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])] # 得到每个bbox与n个anchor的iou的最大值 shape=(k,)
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]
        
        # 返回值 argmax_ious 为每个anchor与所有bbox计算iou得到最大iou值所对应的bbox的index
        #        max_ious 为每个anchor与所有bbox计算iou得到最大的iou值
        #       gt_argmax_ious 为bbox与anchor求iou的最大值对应的anchor的index
        return argmax_ious, max_ious, gt_argmax_ious


def _unmap(data, count, index, fill=0):
    # Unmap a subset of item (data) back to the original set of items (of
    # size count)

    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret


def _get_inside_index(anchor, H, W):
    # Calc indicies of anchors which are located completely inside of the image
    # whose size is speficied.
    index_inside = np.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= H) &
        (anchor[:, 3] <= W)
    )[0]
    return index_inside


class ProposalCreator:

    def __init__(self,
                 parent_model,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16
                 ):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score,
                 anchor, img_size, scale=1.):
        # NOTE: when test, remember
        # faster_rcnn.eval()
        # to set self.traing = False
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        # Convert anchors into proposal via bbox transformations.
        # roi = loc2bbox(anchor, loc)
        # anchor.shape = (9*H*W,4) ,loc.shape = (9*H*W,4)
        # anchor 是基础坐标 loc是偏移量
        roi = loc2bbox(anchor, loc)
        # roi.shape = loc.shape = (9*H*W,4)
        # Clip predicted boxes to image.
        # slice 函数用于进行切片操作，三个参数分别为start end step
        # 分别对roi的x坐标和y坐标做一个clip处理将其限制在0 - img_size之间
        # image_size 是原图像的size，而非特征图的size
        roi[:, slice(0, 4, 2)] = np.clip(
            roi[:, slice(0, 4, 2)], 0, img_size[0])
        roi[:, slice(1, 4, 2)] = np.clip(
            roi[:, slice(1, 4, 2)], 0, img_size[1])

        # Remove predicted boxes with either height or width < threshold.
        min_size = float(self.min_size * scale)
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        keep = np.where((hs >= min_size) & (ws >= min_size))[0] # 得到满足hs>=min_size&& ws>=min_size 的索引
        roi = roi[keep, :] # 保留满足条件的索引所对应的数据
        score = score[keep] # 同样将分类的数据也做对应更改

        # Sort all (proposal, score) pairs by score from highest to lowest.
        # Take top pre_nms_topN (e.g. 6000).
        # ravel() 将数据转换为一维数据  argsort则是对数据进行排序(默认是按从小到大的顺序)，返回的是排序数据的index
        # [::-1] 对list做一个逆序操作 得到从大到小的index排序
        order = score.ravel().argsort()[::-1] 
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]

        # Apply nms (e.g. threshold = 0.7).
        # Take after_nms_topN (e.g. 300).

        # unNOTE: somthing is wrong here!
        # TODO: remove cuda.to_gpu
        keep = nms(
            torch.from_numpy(roi).cuda(),
            torch.from_numpy(score).cuda(),
            self.nms_thresh)
        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        roi = roi[keep.cpu().numpy()]
        return roi
