import os

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from src_yolo_v5.config import VOC_CLASSES, VOC_IMG_MEAN, YOLO_IMG_DIM


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results to filter out overlapping bounding boxes.

    Args:
        prediction (tensor): Output from the YOLOv5 model.
        conf_thres (float): Confidence threshold to filter predictions.
        iou_thres (float): IoU threshold for NMS.
        classes (list, optional): List of classes to keep.
        agnostic (bool, optional): Computes NMS agnostic of class.
        max_det (int, optional): Maximum number of detections per image.

    Returns:
        list of detections, with each detection described by [x1, y1, x2, y2, confidence, class]
    """
    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after

    # Batched NMS
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[xc[xi]]  # confidence
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if agnostic:
            x = torch.cat((box, x[:, 4:5], torch.full((x.shape[0], 1), nc, dtype=torch.long)), 1)
        else:
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        output[xi] = x[i]

    return output

def xywh2xyxy(x):
    """Convert bounding box format from [center_x, center_y, width, height] to [x1, y1, x2, y2]."""
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def predict_image(model, image_name, root_img_directory=""):
    """
    Predict output for a single image using YOLOv5.

    :param model: YOLOv5 model for inference
    :param image_name: image file name e.g. '0000000.jpg'
    :param root_img_directory: root directory where images are stored
    :return: List of lists containing:
        - (x1, y1)
        - (x2, y2)
        - predicted class name
        - image name
        - predicted class probability
    """

    result = []
    image_path = os.path.join(root_img_directory, image_name)
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    # Resize and normalize the image
    img = cv2.resize(image, (YOLO_IMG_DIM, YOLO_IMG_DIM))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0  # Normalize to [0, 1]
    img = img - np.array(VOC_IMG_MEAN, dtype=np.float32) / 255.0

    # Transform
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img).unsqueeze(0)  # Add batch dimension

    # Model inference
    model.eval()
    with torch.no_grad():
        img = img.cuda()  # Send to GPU
        pred = model(img)  # Get predictions

        # Process predictions
        pred = pred[0].cpu()  # Take first item in batch and move to CPU
        pred = non_max_suppression(pred)  # Apply NMS

        # Format output
        for det in pred:  # Iterate over detections
            if det is not None and len(det):
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    result.append([(x1, y1), (x2, y2), VOC_CLASSES[int(cls)], image_name, conf.item()])

    return result
