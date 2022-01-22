from __future__ import division

from collections import defaultdict
import itertools
import numpy as np
import six

from model.utils.bbox_tools import bbox_iou


def eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
        gt_difficults=None,
        iou_thresh=0.5, use_07_metric=False):

    prec, rec = calc_detection_voc_prec_rec(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        iou_thresh=iou_thresh)

    ap = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)

    return {'ap': ap, 'map': np.nanmean(ap)}


def calc_detection_voc_prec_rec(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
        gt_difficults=None,
        iou_thresh=0.5):

    # 使用内置的iter函数用于生成可迭代对象
    pred_bboxes = iter(pred_bboxes) # shape = (N,R,4)
    pred_labels = iter(pred_labels) # shape (N,R) labels预测的是类别 0 1 2 3 4
    pred_scores = iter(pred_scores) # shape (N,R)
    gt_bboxes = iter(gt_bboxes) # GT bboxes存的是真值框coordinates
    gt_labels = iter(gt_labels) # GT label也是类别0 1 2 3 4
    
    if gt_difficults is None:
        gt_difficults = itertools.repeat(None)
    else:
        gt_difficults = iter(gt_difficults)

    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)

    for pred_bbox, pred_label, pred_score, gt_bbox, gt_label, gt_difficult in \
            six.moves.zip(
                pred_bboxes, pred_labels, pred_scores,
                gt_bboxes, gt_labels, gt_difficults):

        if gt_difficult is None:
            gt_difficult = np.zeros(gt_bbox.shape[0], dtype=bool)
        # np.unique用于去除重复排序的数据，并排序输出
        # 对预测框的label和真值框label做了一个cat处理，
        # 之后在做一个unique，将重复的去除
        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            # 再与unique后的数据逐个进行比较
            # pred_mask_l得到的是一个bool类型的 序列
            # 这里实际上是在做什么,得到unique后的数据，实际就是预测框和真值框 的类别的集合
            # pred_mask_l 就是得到当前类别的掩码
            pred_mask_l = pred_label == l
            # pred_bbox_l 就是l这个类别的坐标信息
            pred_bbox_l = pred_bbox[pred_mask_l]
            # pred_score_l 就是l这个类别的得分情况 置信度
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            # 对置信度降序排序
            order = pred_score_l.argsort()[::-1]
            # 按排序次序对坐标信息和置信度排序
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]
            
            # gt_mask_l就是真值框 当前类别l的掩码
            gt_mask_l = gt_label == l
            
            # 得到当前类别的真值 gt_bbox信息
            gt_bbox_l = gt_bbox[gt_mask_l]
            # 得到当前类别的difficult信息
            gt_difficult_l = gt_difficult[gt_mask_l]

            # 对当前这个类别除去difficult 之后的和为真值框的个数 gt_box
            n_pos[l] += np.logical_not(gt_difficult_l).sum()  
            score[l].extend(pred_score_l) # l这个类别的预测置信度加入score

            # 如果pred_bbox_l的长度为0,说明l这个类别只在真值框中出现
            if len(pred_bbox_l) == 0:
                continue
            # 如果gt_bbox_l的长度为0,说明l这个类别只在预测框中出现
            if len(gt_bbox_l) == 0:
                # 就将len_pred_bbox_l个预测框 的match设置为0
                # match 主要是用于标注,哪些是真值,哪些是背景
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VOC evaluation follows integer typed bounding boxes.
            # 这里将y2 x2 +1
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1
            
            # 将预测框与真值框计算iou 得到矩阵 shape=(len_pred_bbox_l,len_gt_bbox_l)
            iou = bbox_iou(pred_bbox_l, gt_bbox_l)
            # 每一行最大的iou的列下标,也就是每个预测框对应的真值框index
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            # 将最大iou小于阈值的index对应的gt设置为-1
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                # 逐个对真值box进行处理
                if gt_idx >= 0:
                    # gt_index >=0 说明iou阈值达标
                    if gt_difficult_l[gt_idx]:
                        # 如果当前索引对应的真值box属于difficult 
                        match[l].append(-1) #match 追加-1 
                    else:
                        if not selec[gt_idx]:
                            # 如果当前真值框没有参与过计算
                            match[l].append(1) # match追加1
                        else:
                            match[l].append(0) # 否则追加0
                    selec[gt_idx] = True # flag =True表示当前真值框已经参与过计算
                else:
                    # 对于iou阈值不达标的match 添加0
                    match[l].append(0) 
                # match[l]数据的顺序,与gt_index有关 而gt_index 对应着预测框的顺序
    
    """
    这里总结一下:
    * match = 1  表示当前预测框为TP
    * match = 0  表示当前预测框为FP FP的情况就是iou阈值不达标或者对应的真框参与过计算
    * match = -1 表示当前预测框为difficult
    """
    for iter_ in (
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels, gt_difficults):
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same.')
    # n_pos 作为一个dict 
    # n_pos key 是 unique(真值框和预测框)
    n_fg_class = max(n_pos.keys()) + 1 #这里为啥要取max
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec


def calc_detection_voc_ap(prec, rec, use_07_metric=False):
    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in six.moves.range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap
