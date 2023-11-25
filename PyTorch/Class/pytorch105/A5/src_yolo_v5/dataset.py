import os
import random

import cv2
import numpy as np

import torch
import torch.utils.data as DataLoader
import torchvision.transforms as transforms

from src_yolo_v5.config import VOC_IMG_MEAN


class VocDetectorDataset640(DataLoader.Dataset):
    image_size = 640

    def __init__(
        self,
        root_img_dir,
        dataset_file,
        train,
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
        # Load image and bounding boxes
        fname = self.fnames[idx]
        img = cv2.imread(os.path.join(self.root, fname))
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()

        # Data augmentation (if training)
        if self.train and self.preproc:
            img, boxes = self.random_flip(img, boxes)
            img, boxes = self.random_scale(img, boxes)
            img, boxes, labels = self.random_shift(img, boxes, labels)
            img, boxes, labels = self.random_crop(img, boxes, labels)

        # Resize image and adjust bounding boxes to YOLOv5 input size
        img, boxes = self.resize_image_and_boxes(img, boxes, self.image_size)

        # Normalize bounding box coordinates and convert image to RGB
        h, w, _ = img.shape
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply transformations
        for t in self.transform:
            img = t(img)

        # Encoding and adjusting target format
        if self.encode_target:
            targets = self.encoder(boxes, labels)  # Targets for each scale
            # Adjust the targets to match model output
            targets = [target.permute(2, 0, 1, 3) for target in targets]  # Transpose to [B, S, S, 5 + C]
        else:
            targets = boxes[:, 0:4].clone()

        if self.return_image_id:
            return img, targets, fname

        return img, *targets  # Unpack the adjusted targets for each scale

    def resize_image_and_boxes(self, img, boxes, new_size):
        """Resize image and adjust bounding boxes for the new image size."""
        old_size = img.shape[:2]
        ratio = float(new_size) / max(old_size)
        new_shape = tuple([round(x * ratio) for x in old_size])
        dw = (new_size - new_shape[1]) / 2  # width padding
        dh = (new_size - new_shape[0]) / 2  # height padding
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)

        # Resize image
        img = cv2.resize(img, (new_shape[1], new_shape[0]))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))

        # Adjust bounding boxes
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * ratio + dw
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * ratio + dh

        return img, boxes

    def __len__(self):
        return self.num_samples

    def encoder(self, boxes, labels):
        """
        Adjusted encoder for YOLOv5 to output target tensors for three different scales: 80x80, 40x40, and 20x20.
        Each scale has a depth of 25 (4 for bbox, 1 for objectness score, and 20 for class probabilities).
        """
        S = [80, 40, 20]  # Three scales
        B = 3  # Number of anchors
        C = 20  # Number of classes for VOC2007
        targets = [torch.zeros((s, s, B, 5 + C)) for s in S]  # 5 for bbox (4) and objectness score (1)

        for box, label in zip(boxes, labels):
            # Normalize box coordinates
            gx, gy, gw, gh = self.normalize_box(box)

            # Assign targets for each scale
            for i, s in enumerate(S):
                # Compute grid cell indices
                cell_size = 1 / s
                grid_x, grid_y = int(gx // cell_size), int(gy // cell_size)

                # Assign target
                target = targets[i]
                target[grid_y, grid_x, :, :4] = torch.tensor([gx, gy, gw, gh])
                target[grid_y, grid_x, :, 4] = 1  # Objectness score
                target[grid_y, grid_x, :, 5 + label] = 1  # Class label

        return targets

    def normalize_box(self, box):
        """
        Normalize bounding box coordinates to be in range [0, 1].
        """
        x1, y1, x2, y2 = box
        gw, gh = x2 - x1, y2 - y1  # width and height of box
        gx, gy = x1 + gw / 2, y1 + gh / 2  # center of box
        return gx, gy, gw, gh

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
