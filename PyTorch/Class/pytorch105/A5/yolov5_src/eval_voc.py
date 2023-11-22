import sys
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from src.predict import *

from src.config import VOC_CLASSES


# https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
def voc_ap(rec, prec, use_07_metric=False):
    """ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(
    preds, target, VOC_CLASSES=VOC_CLASSES, threshold=0.5, use_07_metric=False
):
    """
    preds {'cat':[[image_id,confidence,x1,y1,x2,y2],...],'dog':[[],...]}
    target {(image_id,class):[[],]}
    """
    aps = []
    for i, class_ in enumerate(VOC_CLASSES):
        pred = preds[class_]  # [[image_id,confidence,x1,y1,x2,y2],...]
        if len(pred) == 0:  # No predictions made for this class
            ap = 0.0
            print(
                "---class {} ap {}--- (no predictions for this class)".format(
                    class_, ap
                )
            )
            aps += [ap]
            continue
        image_ids = [x[0] for x in pred]
        confidence = np.array([float(x[1]) for x in pred])
        BB = np.array([x[2:] for x in pred])
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        npos = 0.0
        for (key1, key2) in target:
            if key2 == class_:
                npos += len(target[(key1, key2)])
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d, image_id in enumerate(image_ids):
            bb = BB[d]
            if (image_id, class_) in target:
                BBGT = target[(image_id, class_)]
                for bbgt in BBGT:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(bbgt[0], bb[0])
                    iymin = np.maximum(bbgt[1], bb[1])
                    ixmax = np.minimum(bbgt[2], bb[2])
                    iymax = np.minimum(bbgt[3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
                    ih = np.maximum(iymax - iymin + 1.0, 0.0)
                    inters = iw * ih

                    union = (
                        (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                        + (bbgt[2] - bbgt[0] + 1.0) * (bbgt[3] - bbgt[1] + 1.0)
                        - inters
                    )
                    if union == 0:
                        print(bb, bbgt)

                    overlaps = inters / union
                    if overlaps > threshold:
                        tp[d] = 1
                        BBGT.remove(bbgt)  # bbox has already been used
                        if len(BBGT) == 0:
                            del target[
                                (image_id, class_)
                            ]  # delete things that don't have bbox
                        break
                fp[d] = 1 - tp[d]
            else:
                fp[d] = 1
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
        print("---class {} ap {}---".format(class_, ap))
        aps += [ap]
    print("---map {}---".format(np.mean(aps)))
    return aps


def evaluate(model, val_dataset_file, img_root, val_loader=None):
    targets = defaultdict(list)
    preds = defaultdict(list)
    image_list = []  # image path list

    f = open(val_dataset_file)
    lines = f.readlines()
    file_list = []
    for line in lines:
        splited = line.strip().split()
        file_list.append(splited)
    f.close()

    # Collect target predictions for test set
    for index, image_file in enumerate(file_list):
        image_id = image_file[0]

        image_list.append(image_id)
        num_obj = (len(image_file) - 1) // 5
        for i in range(num_obj):
            x1 = int(image_file[1 + 5 * i])
            y1 = int(image_file[2 + 5 * i])
            x2 = int(image_file[3 + 5 * i])
            y2 = int(image_file[4 + 5 * i])
            c = int(image_file[5 + 5 * i])
            class_name = VOC_CLASSES[c]
            targets[(image_id, class_name)].append([x1, y1, x2, y2])

    print("---Evaluate model on test samples---")
    sys.stdout.flush()
    model.eval()
    for image_path in tqdm(image_list):
        result = predict_image(model, image_path, root_img_directory=img_root)
        for (
            (x1, y1),
            (x2, y2),
            class_name,
            image_id,
            prob,
        ) in result:  # image_id is actually image_path
            preds[class_name].append([image_id, prob, x1, y1, x2, y2])

    aps = voc_eval(preds, targets, VOC_CLASSES=VOC_CLASSES)
    return aps

def test_evaluate(model, test_dataset_file, img_root, test_loader=None):
    targets = defaultdict(list)
    preds = defaultdict(list)
    preds_submission = defaultdict(list)
    image_list = []  # image path list

    f = open(test_dataset_file)
    lines = f.readlines()
    file_list = []
    for line in lines:
        splited = line.strip().split()
        file_list.append(splited)
    f.close()

    # Collect target predictions for test set
    for index, image_file in enumerate(file_list):
        image_id = image_file[0]

        image_list.append(image_id)
        num_obj = (len(image_file) - 1) // 5
        for i in range(num_obj):
            x1 = int(image_file[1 + 5 * i])
            y1 = int(image_file[2 + 5 * i])
            x2 = int(image_file[3 + 5 * i])
            y2 = int(image_file[4 + 5 * i])
            c = int(image_file[5 + 5 * i])
            class_name = VOC_CLASSES[c]
            targets[(image_id, class_name)].append([x1, y1, x2, y2])

    print("---Evaluate model on test samples---")
    sys.stdout.flush()
    model.eval()
    for image_path in tqdm(image_list):
        result = predict_image(model, image_path, root_img_directory=img_root)
        for (
            (x1, y1),
            (x2, y2),
            class_name,
            image_id,
            prob,
        ) in result:  # image_id is actually image_path
            preds[class_name].append([image_id, prob, x1, y1, x2, y2])
            preds_submission[image_id].append([class_name, prob, x1, y1, x2, y2])

    return preds_submission
