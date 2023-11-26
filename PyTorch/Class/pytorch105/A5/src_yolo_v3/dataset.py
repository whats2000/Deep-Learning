# Copyright (c) 2023 Aladdin Persson
# The following code is derived from the YOLOv3 implementation by Aladdin Persson available at
# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLOv3/
"""
Creates a Pytorch dataset to load the Pascal VOC & MS COCO datasets
"""

import src_yolo_v3.config as config
import numpy as np
import os
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from src_yolo_v3.utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    def __init__(
        self,
        img_dir,
        label_file,
        anchors,
        image_size=416,
        S=[13, 26, 52],
        C=20,
        transform=None,
    ):
        self.img_dir = img_dir
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5

        # Read data from the label file
        self.labels = []
        with open(label_file, 'r') as file:
            self.labels = file.readlines()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        line = self.labels[index].strip().split()
        img_path = os.path.join(self.img_dir, line[0])
        image = Image.open(img_path).convert("RGB")

        # Process bounding boxes
        boxes = []
        for i in range(1, len(line), 5):
            x1, y1, x2, y2, class_label = map(int, line[i:i + 5])
            x = ((x1 + x2) / 2) / image.width  # Normalized Center x coordinate
            y = ((y1 + y2) / 2) / image.height  # Normalized Center y coordinate
            w = (x2 - x1) / image.width  # Normalized Width
            h = (y2 - y1) / image.height  # Normalized Height
            boxes.append([x, y, w, h, class_label])

        image = np.array(image)

        if self.transform:
            augmentations = self.transform(image=image, bboxes=boxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3  # each scale should have one anchor
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)  # which cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        return image, tuple(targets)


def test():
    anchors = config.ANCHORS

    transform = config.train_transforms

    dataset = YOLODataset(
        img_dir='C:/Users/eddie/GitHub/Deep-Learning/PyTorch/Class/pytorch105/A5/data/VOCdevkit_2007/VOC2007/JPEGImages',
        label_file='C:/Users/eddie/GitHub/Deep-Learning/PyTorch/Class/pytorch105/A5/data/voc2007train.txt',
        S=[13, 26, 52],
        anchors=anchors,
        transform=transform,
    )
    S = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for x, y in loader:
        boxes = []

        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            print(anchor.shape)
            print(y[i].shape)
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]
        boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
        print(boxes)
        plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes)


if __name__ == "__main__":
    test()