import numpy as np

import six
from six import __init__


def loc2bbox(src_bbox, loc):
    if src_bbox.shape[0] == 0:
        return np.zeros((0, 4), dtype=loc.dtype)

    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False) #这句在干嘛？

    src_height = src_bbox[:, 2] - src_bbox[:, 0]  #p_h = p_ymax-p_ymin
    src_width = src_bbox[:, 3] - src_bbox[:, 1]   #p_w = p_xmax-p_xmin
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height #中心y坐标
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width  #中心x坐标
    
    # [:,0::4]这种切片的方法与[:,0]这种方法的区别在于 前者会保留第二维 而后者不会保留第二维
    # [:,0].shape ->(n) ,[:,0::4].shape -> (n,1)
    # dy dx dh dw 这也是需要学习的参数
    dy = loc[:, 0::4] # (H*W*9,1)
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]
    
    # 通过公式 center_x = source_x + dx*source_width 平移
    #         center_y = source_y + dy*source_height
    #         dst_h = e^dh * source_height 缩放
    #         dst_w = e^dw * source_width
    # 
    ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
    ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]
    h = np.exp(dh) * src_height[:, np.newaxis]
    w = np.exp(dw) * src_width[:, np.newaxis]

    # 通过得到的ctr_x,ctr_y,h,w 计算得到 box的参数
    dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h 
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w

    return dst_bbox


def bbox2loc(src_bbox, dst_bbox):

    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = src_bbox[:, 0] + 0.5 * height
    ctr_x = src_bbox[:, 1] + 0.5 * width

    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

    eps = np.finfo(height.dtype).eps # 防止出现0或负数
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)

    loc = np.vstack((dy, dx, dh, dw)).transpose()
    return loc


def bbox_iou(bbox_a, bbox_b):
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    # top left
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    # return shape is (N,K)
    # 其中(n,k)位置表示在bbox_a 中的第n个box与在bbox_b 中的第k个box之间的iou
    return area_i / (area_a[:, None] + area_b - area_i)


def __test():
    pass


if __name__ == '__main__':
    __test()


def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],
                         anchor_scales=[8, 16, 32]):
    py = base_size / 2.  #8.0
    px = base_size / 2.

    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4),
                           dtype=np.float32)  #  shape is (9,4)
    for i in six.moves.range(len(ratios)):
        for j in six.moves.range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i]) # 16 * scale_8 * sqrt(0.5)
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.
    return anchor_base
    # anchor_base.shape = (9,4)
