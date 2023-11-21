import os
import random

import cv2
import numpy as np

import torch
import torch.utils.data as DataLoader
import torchvision.transforms as transforms

from src.config import VOC_IMG_MEAN


class VocDetectorDataset(DataLoader.Dataset):
    image_size = 448

    def __init__(
        self,
        root_img_dir,
        dataset_file,
        train,
        S,
        preproc=True,
        return_image_id=False,
        encode_target=True,
    ):
        print("Initializing dataset")
        self.root = root_img_dir
        self.train = train
        self.transform = [transforms.ToTensor()]
        self.fnames = []
        self.boxes = []
        self.labels = []
        self.mean = VOC_IMG_MEAN
        self.S = S

        self.return_image_id = return_image_id
        self.encode_target = encode_target

        with open(dataset_file) as f:
            lines = f.readlines()

        for line in lines:
            split_line = line.strip().split()
            self.fnames.append(split_line[0])
            num_boxes = (len(split_line) - 1) // 5
            box = []
            label = []
            for i in range(num_boxes):
                x1 = float(split_line[1 + 5 * i])
                y1 = float(split_line[2 + 5 * i])
                x2 = float(split_line[3 + 5 * i])
                y2 = float(split_line[4 + 5 * i])
                c = split_line[5 + 5 * i]
                box.append([x1, y1, x2, y2])
                label.append(int(c) + 1)
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))
        self.num_samples = len(self.boxes)

        self.preproc = preproc

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.imread(os.path.join(self.root + fname))

        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()

        if self.train and self.preproc:
            img, boxes = self.random_flip(img, boxes)
            img, boxes = self.random_scale(img, boxes)
            img, boxes, labels = self.random_shift(img, boxes, labels)
            img, boxes, labels = self.random_crop(img, boxes, labels)

        h, w, _ = img.shape
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = cv2.cvtColor(
            img, cv2.COLOR_BGR2RGB
        )  # because pytorch pretrained model use RGB
        img = img - np.array(
            self.mean, dtype=np.float32
        )  # subtract dataset mean image (in RGB format)

        if self.encode_target:
            target_boxes, target_cls, has_object_map = self.encoder(
                boxes, labels
            )  # SxSx(B*5+C)
        else:
            target = list(boxes[idx][:, 0:4]).clone()

        for t in self.transform:
            img = t(img)

        if self.return_image_id:
            return img, target, fname

        return img, target_boxes, target_cls, has_object_map

    def __len__(self):
        return self.num_samples

    def encoder(self, boxes, labels):
        """
        This function takes as input bounding boxes and corresponding labels for a particular image
        sample and outputs a target tensor of size SxSx(5xB+C)

        boxes (tensor) [[x1,y1,x2,y2],[]]
        labels (tensor) [...]
        return SxSx(5xB+C) (14x14x30 in our case)
        """
        grid_num = self.S
        target = torch.zeros((grid_num, grid_num, 25))
        cell_size = 1.0 / grid_num
        wh = boxes[:, 2:] - boxes[:, :2]
        center_xy_all = (boxes[:, 2:] + boxes[:, :2]) / 2
        for i in range(center_xy_all.size()[0]):
            center_xy = center_xy_all[i]
            ij = (center_xy / cell_size).ceil() - 1
            # confidence represents iou between predicted and ground truth
            target[int(ij[1]), int(ij[0]), 4] = 1  # confidence of box 1
            target[int(ij[1]), int(ij[0]), int(labels[i]) + 4] = 1
            xy = ij * cell_size  # coordinates of upper left corner
            delta_xy = (center_xy - xy) / cell_size
            target[int(ij[1]), int(ij[0]), 2:4] = wh[i]
            target[int(ij[1]), int(ij[0]), :2] = delta_xy

        target_boxes = target[:, :, :4]
        has_object_map = (target[:, :, 4:5]) > 0
        has_object_map = has_object_map.squeeze()
        target_cls = target[:, :, 5:]

        return target_boxes, target_cls, has_object_map

    def random_shift(self, img, boxes, labels):
        # Augment data with a small translational shift
        center = (boxes[:, 2:] + boxes[:, :2]) / 2
        if random.random() < 0.5:
            height, width, c = img.shape
            after_shfit_image = np.zeros((height, width, c), dtype=img.dtype)
            after_shfit_image[:, :, :] = (104, 117, 123)  # bgr
            shift_x = random.uniform(-width * 0.2, width * 0.2)
            shift_y = random.uniform(-height * 0.2, height * 0.2)

            # translate image by a shift factor
            if shift_x >= 0 and shift_y >= 0:
                after_shfit_image[int(shift_y) :, int(shift_x) :, :] = img[
                    : height - int(shift_y), : width - int(shift_x), :
                ]
            elif shift_x >= 0 and shift_y < 0:
                after_shfit_image[: height + int(shift_y), int(shift_x) :, :] = img[
                    -int(shift_y) :, : width - int(shift_x), :
                ]
            elif shift_x < 0 and shift_y >= 0:
                after_shfit_image[int(shift_y) :, : width + int(shift_x), :] = img[
                    : height - int(shift_y), -int(shift_x) :, :
                ]
            elif shift_x < 0 and shift_y < 0:
                after_shfit_image[
                    : height + int(shift_y), : width + int(shift_x), :
                ] = img[-int(shift_y) :, -int(shift_x) :, :]

            shift_xy = torch.FloatTensor([[int(shift_x), int(shift_y)]]).expand_as(
                center
            )
            center = center + shift_xy
            mask1 = (center[:, 0] > 0) & (center[:, 0] < width)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < height)
            mask = (mask1 & mask2).view(-1, 1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if len(boxes_in) == 0:
                return img, boxes, labels
            box_shift = torch.FloatTensor(
                [[int(shift_x), int(shift_y), int(shift_x), int(shift_y)]]
            ).expand_as(boxes_in)
            boxes_in = boxes_in + box_shift
            labels_in = labels[mask.view(-1)]
            return after_shfit_image, boxes_in, labels_in
        return img, boxes, labels

    def random_scale(self, img, boxes):
        # Augment data with a random scaling of image
        scale_upper_bound, scale_lower_bound = (0.8, 1.2)
        if random.random() < 0.5:
            scale = random.uniform(scale_upper_bound, scale_lower_bound)
            height, width, c = img.shape
            img = cv2.resize(img, (int(width * scale), height))
            scale_tensor = torch.FloatTensor([[scale, 1, scale, 1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            return img, boxes
        return img, boxes

    def random_crop(self, img, boxes, labels):
        # Augment data with a random crop of image sample
        if random.random() < 0.5:
            center = (boxes[:, 2:] + boxes[:, :2]) / 2
            height, width, c = img.shape
            h = random.uniform(0.6 * height, height)
            w = random.uniform(0.6 * width, width)
            x = random.uniform(0, width - w)
            y = random.uniform(0, height - h)
            x, y, h, w = int(x), int(y), int(h), int(w)

            center = center - torch.FloatTensor([[x, y]]).expand_as(center)
            mask1 = (center[:, 0] > 0) & (center[:, 0] < w)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < h)
            mask = (mask1 & mask2).view(-1, 1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if len(boxes_in) == 0:
                return img, boxes, labels
            box_shift = torch.FloatTensor([[x, y, x, y]]).expand_as(boxes_in)

            boxes_in = boxes_in - box_shift
            boxes_in[:, 0] = boxes_in[:, 0].clamp_(min=0, max=w)
            boxes_in[:, 2] = boxes_in[:, 2].clamp_(min=0, max=w)
            boxes_in[:, 1] = boxes_in[:, 1].clamp_(min=0, max=h)
            boxes_in[:, 3] = boxes_in[:, 3].clamp_(min=0, max=h)

            labels_in = labels[mask.view(-1)]
            img_cropped = img[y : y + h, x : x + w, :]
            return img_cropped, boxes_in, labels_in
        return img, boxes, labels

    def random_flip(self, im, boxes):
        # Augment data with a random horizontal image flip
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h, w, _ = im.shape
            xmin = w - boxes[:, 2]
            xmax = w - boxes[:, 0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
            return im_lr, boxes
        return im, boxes

    def subtract_mean(self, im, mean):
        mean = np.array(mean, dtype=np.float32)
        im = im - mean
        return im
