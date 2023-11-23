import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import patches, pyplot as plt
from torchvision.ops import nms

from src_yolo_v5 import config
from src_yolo_v5.config import YOLO_IMG_DIM

# Copyright (c) 2023 Alessandro Mondin
# The following code is derived from the YOLOv5 implementation by Alessandro Mondin available at
# https://github.com/AlessandroMondin/YOLOV5m
def cells_to_bboxes(predictions, anchors, strides, is_pred=False, to_list=True):
    """
    Convert predictions into bboxes.
    Args:
        predictions: Predictions from the model
        anchors: Anchors Boxes for the model
        strides: Strides in head
        is_pred: Is this prediction for prediction or training
        to_list: Convert the output to a list

    Returns:
        Converted boxes
    """
    num_out_layers = len(predictions)
    grid = [torch.empty(0) for _ in range(num_out_layers)]  # initialize
    anchor_grid = [torch.empty(0) for _ in range(num_out_layers)]  # initialize

    all_bboxes = []
    for i in range(num_out_layers):
        bs, naxs, ny, nx, _ = predictions[i].shape
        stride = strides[i]
        grid[i], anchor_grid[i] = make_grids(anchors, naxs, ny=ny, nx=nx, stride=stride, i=i)
        if is_pred:
            # formula here: https://github.com/ultralytics/yolov5/issues/471
            # xy, wh, conf = predictions[i].sigmoid().split((2, 2, 80 + 1), 4)
            layer_prediction = predictions[i].sigmoid()
            obj = layer_prediction[..., 4:5]
            xy = (2 * (layer_prediction[..., 0:2]) + grid[i] - 0.5) * stride
            wh = ((2 * layer_prediction[..., 2:4]) ** 2) * anchor_grid[i]
            best_class = torch.argmax(layer_prediction[..., 5:], dim=-1).unsqueeze(-1)

        else:
            predictions[i] = predictions[i].to('cuda', non_blocking=True)
            obj = predictions[i][..., 4:5]
            xy = (predictions[i][..., 0:2] + grid[i]) * stride
            wh = predictions[i][..., 2:4] * stride
            best_class = predictions[i][..., 5:6]

        scale_bboxes = torch.cat((best_class, obj, xy, wh), dim=-1).reshape(bs, -1, 6)

        all_bboxes.append(scale_bboxes)

    return torch.cat(all_bboxes, dim=1).tolist() if to_list else torch.cat(all_bboxes, dim=1)


def make_grids(anchors, naxs, stride, nx=20, ny=20, i=0):
    x_grid = torch.arange(nx)
    x_grid = x_grid.repeat(ny).reshape(ny, nx)

    y_grid = torch.arange(ny).unsqueeze(0)
    y_grid = y_grid.T.repeat(1, nx).reshape(ny, nx)

    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    xy_grid = xy_grid.expand(1, naxs, ny, nx, 2).to('cuda', non_blocking=True)
    anchor_grid = (anchors[i] * stride).reshape((1, naxs, 1, 1, 2)).expand(1, naxs, ny, nx, 2)

    return xy_grid, anchor_grid

def non_max_suppression(batch_bboxes, iou_threshold, threshold, max_detections=300, to_list=True):
    """
    Non-max suppression for batched boxes
    Args:
        batch_bboxes: Batch Bounding Boxes
        iou_threshold: Intersection over union threshold
        threshold: Classification threshold
        max_detections: Max detections per image
        to_list: Convert the output to a list

    Returns:
        Non-maximum suppressed bounding boxes
    """

    bboxes_after_nms = []
    for boxes in batch_bboxes:
        boxes = torch.masked_select(boxes, boxes[..., 1:2] > threshold).reshape(-1, 6)

        # from xywh to x1y1x2y2

        boxes[..., 2:3] = boxes[..., 2:3] - (boxes[..., 4:5] / 2)
        boxes[..., 3:4] = boxes[..., 3:4] - (boxes[..., 5:] / 2)
        boxes[..., 5:6] = boxes[..., 5:6] + boxes[..., 3:4]
        boxes[..., 4:5] = boxes[..., 4:5] + boxes[..., 2:3]

        indices = nms(boxes=boxes[..., 2:] + boxes[..., 0:1], scores=boxes[..., 1], iou_threshold=iou_threshold)
        boxes = boxes[indices]

        if boxes.shape[0] > max_detections:
            boxes = boxes[:max_detections, :]

        bboxes_after_nms.append(
            boxes.tolist() if to_list else boxes
        )

    return bboxes_after_nms if to_list else torch.cat(bboxes_after_nms, dim=0)

def plot_image(image, boxes, labels=config.COCO):
    """
    Plots predicted bounding boxes on the image
    """
    cmap = plt.get_cmap("tab20b")
    class_labels = labels
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    orig_h, orig_w, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the original image
    ax.imshow(image)

    # Scale ratio
    scale_x = orig_w / YOLO_IMG_DIM
    scale_y = orig_h / YOLO_IMG_DIM

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        bbox = box[2:]

        # Scale box to original image size
        upper_left_x = max(bbox[0] * scale_x, 0)
        upper_left_y = max(bbox[1] * scale_y, 0)
        width = (bbox[2] - bbox[0]) * scale_x
        height = (bbox[3] - bbox[1]) * scale_y

        rect = patches.Rectangle(
            (upper_left_x, upper_left_y),
            width,
            height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )

        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            upper_left_x,
            upper_left_y,
            s=f"{class_labels[int(class_pred)]}: {box[1]:.2f}",
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )

    plt.show()
# End of code derived from the YOLOv5m implementation by Alessandro Mondin

def predict_image_yolo_v5(model, image_name, root_img_directory=""):
    """
    Predict output for a single image using YOLOv5.

    :param model: YOLOv5 model for inference
    :param image_name: image file name e.g. '0000000.jpg'
    :param root_img_directory: root directory where images are stored
    :return: List of lists containing:
        - predicted class id
        - predicted class probability
        - x
        - y
        - w
        - h
    """
    result = []
    image_path = os.path.join(root_img_directory, image_name)
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    # Resize and normalize the image
    img = cv2.resize(image, (YOLO_IMG_DIM, YOLO_IMG_DIM))

    # Transform to tensor
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img).unsqueeze(0).float()  # Add batch dimension

    # Model inference
    model.eval()
    with torch.no_grad():
        img = img.cuda()  # Send to GPU
        predict_list = model(img)  # Get predictions

    # Post-process the predictions
    boxes = cells_to_bboxes(predict_list, model.head.anchors, model.head.stride, is_pred=True, to_list=False)

    boxes = non_max_suppression(boxes, iou_threshold=0.45, threshold=0.25, to_list=False)

    return boxes
