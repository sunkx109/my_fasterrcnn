import torch
from torchvision import transforms
from skimage import transform as sktsf
import numpy as np
import random
from utils.config import opt
from data.voc_dataset import VOCBboxDataset


def preprocess(img, min_size=600, max_size=1000):
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.0
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    img = normalize(torch.from_numpy(img))
    return img.numpy()


def resize_bbox(bbox, in_size, out_size):
    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox[:, 0] = x_scale * bbox[:, 0]
    bbox[:, 2] = x_scale * bbox[:, 2]
    bbox[:, 1] = y_scale * bbox[:, 1]
    bbox[:, 3] = y_scale * bbox[:, 3]
    return bbox

def random_flip(
    img, y_random=False, x_random=False, return_param=False
):
    y_flip, x_flip = False, False
    if y_random:
        y_flip = random.choice([True, False])
    if x_random:
        x_flip = random.choice([True, False])

    if y_flip:
        img = img[:, ::-1, :]
    if x_flip:
        img = img[:, :, ::-1]

    if return_param:
        return img, {"y_flip": y_flip, "x_flip": x_flip}
    else:
        return img

def flip_bbox(bbox, size, y_flip=False, x_flip=False):
    H, W = size
    if y_flip:
        y_max = H - bbox[:, 1]
        y_min = H - bbox[:, 3]
        bbox[:, 1] = y_min
        bbox[:, 3] = y_max
    if x_flip:
        x_max = W - bbox[:, 0]
        x_min = W - bbox[:, 2]
        bbox[:, 0] = x_min
        bbox[:, 2] = x_max
    return bbox

class Transform(object):
    
    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip
        img, params = random_flip(
            img, x_random=True, return_param=True)
        bbox = flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale


class TrainDataSet:
    def __init__(self,opt):
        self.opt = opt
        self.dataset = VOCBboxDataset(opt.voc_data_dir)
        self.transform = Transform(opt.min_size, opt.max_size)
    
    def __len__(self):
        return self.dataset.__len__()
    
    def __getitem__(self,index):
        ori_img, bbox, label, difficult = self.dataset.get_example(index)
        img, bbox, label, scale = self.transform((ori_img, bbox, label))
        return img.copy(), bbox.copy(), label.copy(), scale
    
        

class TestDataSet:
    def __init__(self,opt,split="test",use_difficult=True):
        self.opt = opt
        self.dataset = VOCBboxDataset(
            opt.voc_data_dir, split=split, use_difficult=use_difficult
        )
        
    def __len__(self):
        return self.dataset.__len__() 
    
    def __getitem__(self, index):
        ori_img, bbox, label ,difficult= self.dataset.get_example(index)
        img = preprocess(ori_img)
        return img, ori_img.shape[1:], bbox, label,difficult  
        
        