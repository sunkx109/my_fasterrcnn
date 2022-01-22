import torch
from torch._C import dtype
from utils.config import opt
from data.dataset import TrainDataSet, TestDataSet,preprocess
from tqdm import tqdm
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.eval_tool import *
from PIL import Image
import cv2


def eval(dataloader, faster_rcnn):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for index, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(
        enumerate(dataloader)
    ):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(
            imgs, [sizes]
        )
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_

    result = eval_detection_voc(
        pred_bboxes,
        pred_labels,
        pred_scores,
        gt_bboxes,
        gt_labels,
        gt_difficults,
        use_07_metric=True,
    )
    return result


def train():
    dataset = TrainDataSet(opt)
    print("load data")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,  # pin_memory=True,
        num_workers=opt.num_workers,
    )
    testset = TestDataSet(opt)
    test_dataloader = torch.utils.data.DataLoader(
        testset,
        batch_size=1,
        num_workers=opt.test_num_workers,
        shuffle=False,
        pin_memory=True,
    )
    faster_rcnn = FasterRCNNVGG16()
    print("model construct completed")
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()

    best_map = 0
    lr_ = opt.lr
    for epoch in range(opt.epoch):
        for index, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            # scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            loss = trainer.train_step(img, bbox, label, scale)
            # if index % 30 == 0:
            #     print(
            #         "epoch",
            #         epoch,
            #         " rpn_loc_loss:",
            #         float(loss[0].data),
            #         "rpn_cls_loss:",
            #         float(loss[1].data),
            #         "roi_loc_loss:",
            #         float(loss[2].data),
            #         "roi_cls_loss:",
            #         float(loss[3].data),
            #         "total_loss:",
            #         float(loss[4].data),
            #     )

        eval_result = eval(test_dataloader, trainer.faster_rcnn)
        print("Epoch", epoch, "  the eval_result is:", eval_result)

        if eval_result["map"] > best_map:
            best_map = eval_result["map"]
            torch.save(
                trainer.faster_rcnn.state_dict(),
                "/home/users/kaixin.sun/object_detection/fasterrcnn/save_pth/model.pt",
            )
            print("save the best map", best_map, " model successfully")
        if epoch == 9:
            state_dict = torch.load(
                "/home/users/kaixin.sun/object_detection/fasterrcnn/save_pth/model.pt"
            )
            trainer.faster_rcnn.load_state_dict(state_dict)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay


def read_image(path, dtype=np.float32, color=True):
    f = Image.open(path)
    try:
        if color:
            img = f.convert("RGB")
        else:
            img = f.convert("P")
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, "close"):
            f.close()

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))

def predict():
    faster_rcnn = FasterRCNNVGG16().cuda()
    state_dict = torch.load(
        "./save_pth/model.pt"
    )
    faster_rcnn.load_state_dict(state_dict)
    img = read_image("./imgs/004585.jpg")
    # ori_H,ori_W = img.shape[1:]
    img = preprocess(img)
    H,W = img.shape[1:]
    # scale = H / ori_H
    img = img[None]
    size = torch.tensor([H,W])
    
    bboxes,labels,scores = faster_rcnn.predict(img,[size])
    print(bboxes)
    print(labels)
    print(scores)       
            
            


if __name__ == "__main__":
    train()
    # predict()
