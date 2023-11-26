import torch
import torch.nn as nn
import torch.nn.functional as F

from src_yolo_v3.utils import intersection_over_union


def compute_iou(box1, box2):
    """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    """
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou


class YoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def xywh2xyxy(self, boxes):
        """
        Parameters:
        boxes: (N,4) representing by x,y,w,h

        Returns:
        boxes: (N,4) representing by x1,y1,x2,y2

        if for a Box b the coordinates are represented by [x, y, w, h] then
        x1, y1 = x/S - 0.5*w, y/S - 0.5*h ; x2,y2 = x/S + 0.5*w, y/S + 0.5*h
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        """
        ### CODE ###
        # Your code here
        # Extracting x, y, width, and height
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        # Normalizing x and y by the grid size (S)
        x, y = x / self.S, y / self.S

        # Converting to corner coordinates
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        # Stacking the coordinates
        converted_boxes = torch.stack([x1, y1, x2, y2], dim=1)

        return converted_boxes

    def find_best_iou_boxes(self, box_pred_list, box_target):
        """
        Parameters:
        box_pred_list : [(tensor) size (M, 5) ...], length of list = B
        box_target : (tensor)  size (N, 4)

        Returns:
        best_iou: (tensor) size (-1, 1)
        best_boxes : (tensor) size (-1, 5), containing the boxes which give the best iou among the two (self.B) predictions

        Hints:
        1) Find the iou's of each of the 2 bounding boxes of each grid cell of each image.
        2) For finding iou's use the compute_iou function
        3) use self.xywh2xyxy() to convert bbox format if necessary,
        4) hint: use torch.diagnoal() on results of compute_iou
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        We perform this transformation to convert the correct coordinates into bounding box coordinates.
        """
        ### CODE ###
        # Your code here
        # Initialize tensors to store the best IOU and corresponding boxes
        best_ious = torch.zeros(box_target.size(0), 1)
        best_boxes = torch.zeros(box_target.size(0), 5)

        # Iterate through all predicted boxes
        for i in range(box_target.size(0)):
            ious = []
            for box_pred in box_pred_list:
                box_pred_xyxy = self.xywh2xyxy(box_pred[:, :4])
                iou = compute_iou(box_pred_xyxy[i].unsqueeze(0), box_target[i].unsqueeze(0))
                ious.append(iou.squeeze())

            # Find the maximum IOU and the corresponding predicted box
            max_iou, best_idx = torch.max(torch.stack(ious), dim=0)
            best_ious[i] = max_iou
            best_boxes[i] = box_pred_list[best_idx][i]

        return best_ious, best_boxes

    def get_class_prediction_loss(self, classes_pred, classes_target, has_object_map):
        """
        Parameters:
        classes_pred : (tensor) size (batch_size, S, S, 20)
        classes_target : (tensor) size (batch_size, S, S, 20)
        has_object_map: (tensor) size (batch_size, S, S)

        Returns:
        class_loss : scalar
        """
        ### CODE ###
        # Your code here
        # Apply the object presence mask to the predictions and targets
        object_mask = has_object_map.unsqueeze(-1).expand_as(
            classes_pred)  # (batch_size, S, S) -> (batch_size, S, S, 20)
        classes_pred = classes_pred[object_mask].view(-1, 20)
        classes_target = classes_target[object_mask].view(-1, 20)

        return F.mse_loss(classes_pred, classes_target, reduction='sum')

    def get_no_object_loss(self, pred_boxes_list, has_object_map):
        """
        Parameters:
        pred_boxes_list: (list) [(tensor) size (Batch, S, S, 5)  for B pred_boxes]
        has_object_map: (tensor) size (Batch, S, S): Mask for cells which contain objects

        Returns:
        loss : scalar

        Hints:
        1) Only compute loss for cell which doesn't contain object
        2) compute loss for all predictions in the pred_boxes_list list
        3) You can assume the ground truth confidence of non-object cells is 0
        """
        ### CODE ###
        # Your code here
        # Invert the object map to focus on cells without objects
        not_object_mask = ~has_object_map

        # Repeat the mask for each bounding box prediction in the list
        repeated_not_object_mask = not_object_mask.unsqueeze(-1).repeat(1, 1, 1, self.B).reshape(-1)

        # Concatenate all prediction confidence scores and reshape
        flat_pred_conf = torch.cat([pred_boxes[..., 4].reshape(-1) for pred_boxes in pred_boxes_list], dim=0)

        # Apply the mask to focus on no-object confidence scores
        pred_conf_no_obj = flat_pred_conf[repeated_not_object_mask]

        # The target confidence for no-object cells is 0
        target_conf_no_obj = torch.zeros_like(pred_conf_no_obj)

        # Compute Mean Squared Error loss
        no_obj_loss = F.mse_loss(pred_conf_no_obj, target_conf_no_obj, reduction='sum')

        return no_obj_loss

    def get_contain_conf_loss(self, box_pred_conf, box_target_conf):
        """
        Parameters:
        box_pred_conf : (tensor) size (-1,1)
        box_target_conf: (tensor) size (-1,1)

        Returns:
        contain_loss : scalar

        Hints:
        The box_target_conf should be treated as ground truth, i.e., no gradient

        """
        ### CODE
        # your code here
        # Use mean squared error loss
        contain_conf_loss = F.mse_loss(box_pred_conf, box_target_conf, reduction='sum')

        return contain_conf_loss

    def get_regression_loss(self, box_pred_response, box_target_response):
        """
        Parameters:
        box_pred_response : (tensor) size (M, 4)
        box_target_response : (tensor) size (M, 4)
        Note : -1 corresponds to ravels the tensor into the dimension specified
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns:
        reg_loss : scalar
        """
        ### CODE
        # your code here
        # Use Mean Squared Error loss for regression
        # Ensure both tensors are on the same device
        if box_pred_response.device != box_target_response.device:
            box_pred_response = box_pred_response.to(box_target_response.device)

        # Compute the Mean Squared Error loss for center coordinates
        center_loss = F.mse_loss(box_pred_response[:, :2], box_target_response[:, :2], reduction='sum')

        # Compute the Mean Squared Error loss for dimensions, comparing square roots to emphasize smaller boxes
        dimension_loss = F.mse_loss(torch.sqrt(box_pred_response[:, 2:4]), torch.sqrt(box_target_response[:, 2:4]),
                                    reduction='sum')

        # Combine the losses
        reg_loss = center_loss + dimension_loss

        return reg_loss

    def forward(self, pred_tensor, target_boxes, target_cls, has_object_map):
        """
        pred_tensor: (tensor) size(N,S,S,Bx5+20=30) N:batch_size
                      where B - number of bounding boxes this grid cell is a part of = 2
                            5 - number of bounding box values corresponding to [x, y, w, h, c]
                                where x - x_coord, y - y_coord, w - width, h - height, c - confidence of having an object
                            20 - number of classes

        target_boxes: (tensor) size (N, S, S, 4): the ground truth bounding boxes
        target_cls: (tensor) size (N, S, S, 20): the ground truth class
        has_object_map: (tensor, bool) size (N, S, S): the ground truth for whether each cell contains an object (True/False)

        Returns:
        loss_dict (dict): with key value stored for total_loss, reg_loss, containing_obj_loss, no_obj_loss and cls_loss
        """
        N = pred_tensor.size(0)
        inv_n = 1.0 / N
        # When you calculate the classification loss, no-object loss, regression loss, contain_object_loss
        # you need to multiply the loss with inv_n. e.g: inv_n * self.get_regression_loss(...)

        # split the pred tensor from an entity to separate tensors:
        # -- pred_boxes_list: a list containing all bbox prediction (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        # -- pred_cls (containing all classification predictions)
        pred_boxes_list = [pred_tensor[:, :, :, i * 5:(i + 1) * 5] for i in range(self.B)]
        pred_cls = pred_tensor[:, :, :, self.B * 5:]  # (N, S, S, 20)

        # compute classification loss
        cls_loss = self.get_class_prediction_loss(pred_cls, target_cls, has_object_map) * inv_n

        # compute no-object loss
        no_obj_loss = self.get_no_object_loss(pred_boxes_list, has_object_map) * inv_n

        # Re-shape boxes in pred_boxes_list and target_boxes to meet the following desires
        # 1) only keep having-object cells
        # 2) vectorize all dimensions except for the last one for faster computation
        pred_boxes_list = [pred_boxes[has_object_map] for pred_boxes in pred_boxes_list]
        target_boxes = target_boxes[has_object_map].view(-1, 4)

        # compute regression loss between the found best bbox and GT bbox for all the cell containing objects
        best_ious, best_boxes = self.find_best_iou_boxes(pred_boxes_list, target_boxes)

        # compute contain_object_loss
        containing_obj_loss = self.get_contain_conf_loss(best_boxes[..., 4].unsqueeze(-1),
                                                         torch.ones_like(best_ious)) * inv_n

        # compute regression loss
        reg_loss = self.get_regression_loss(best_boxes[..., :4], target_boxes[..., :4]) * inv_n

        # compute final loss
        total_loss = cls_loss + self.l_noobj * no_obj_loss + self.l_coord * reg_loss + containing_obj_loss

        # construct return loss_dict
        loss_dict = {
            "total_loss": total_loss,
            "reg_loss": reg_loss,
            "containing_obj_loss": containing_obj_loss,
            "no_obj_loss": no_obj_loss,
            "cls_loss": cls_loss
        }
        return loss_dict

class YoloLossV3(nn.Module):
    def __init__(self):
        super().__init__()
        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 2.5
        self.balance_weight = [1, 1, 1]

    @staticmethod
    def get_no_object_loss(prediction, target, not_object_mask):
        """
        Compute the loss for the cells without any objects
        Args:
            prediction: size(batch, 3, S, S, 25)
            target: size(batch, 3, S, S, 6)
            not_object_mask: size(batch, 3, S, S)

        Returns:
            loss of the cells without any objects
        """
        # Extract the confidence score for the cells without any objects
        pred_conf_not_obj = prediction[..., 0][not_object_mask]
        target_conf_not_obj = target[..., 0][not_object_mask]

        # Compute the loss
        no_object_loss = F.binary_cross_entropy_with_logits(pred_conf_not_obj, target_conf_not_obj)

        return no_object_loss

    @staticmethod
    def get_class_prediction_loss(prediction, target, object_mask):
        """
        Compute the loss for the cells with objects
        Args:
            prediction: size(batch, 3, S, S, 25)
            target: size(batch, 3, S, S, 6)
            object_mask: size(batch, 3, S, S)

        Returns:
            loss of the cells with objects
        """
        # Extract the class predictions for the cells with objects
        pred_class = prediction[..., 5:][object_mask]
        target_class = target[..., 5][object_mask].long()

        # Compute the loss
        class_loss = F.cross_entropy(pred_class, target_class)

        return class_loss

    @staticmethod
    def get_object_loss(prediction, target, object_mask, anchors):
        """
        Compute the loss for the cells with objects
        Args:
            prediction: size(batch, 3, S, S, 25)
            target: size(batch, 3, S, S, 6)
            object_mask: size(batch, 3, S, S)
            anchors: size(batch, 3, 2)

        Returns:
            loss of the cells with objects
        """
        box_preds = torch.cat([F.sigmoid(prediction[..., 1:3]), torch.exp(prediction[..., 3:5]) * anchors], dim=-1)
        ious = intersection_over_union(box_preds[object_mask], target[..., 1:5][object_mask]).detach()
        object_loss = F.mse_loss(F.sigmoid(prediction[..., 0:1][object_mask]), ious * target[..., 0:1][object_mask])

        return object_loss

    @staticmethod
    def get_box_coordination_loss(prediction, target, object_mask, anchors):
        """
        Compute the loss for the cells with objects
        Args:
            prediction: size(batch, 3, S, S, 25)
            target: size(batch, 3, S, S, 6)
            object_mask: size(batch, 3, S, S)
            anchors: size(batch, 3, 2)

        Returns:
            loss of the cells with objects
        """
        # For the cells with objects, compute the loss for the box coordinates
        prediction[..., 1:3] = F.sigmoid(prediction[..., 1:3])

        # For the target zoom in to the cells with objects and extract the box coordinates
        target[..., 3:5] = torch.log((1e-16 + target[..., 3:5] / anchors))

        # Compute the loss
        box_loss = F.mse_loss(prediction[..., 1:5][object_mask], target[..., 1:5][object_mask])

        return box_loss

    def forward(self, predictions_list, target_list, anchors_list):
        """
        Compute the loss for the anchor boxes
        Args:
            predictions_list: [size(batch, 3, 13, 13, 25), size(batch, 3, 26, 26, 25), size(batch, 3, 52, 52, 25)] while 25 is [conf, x, y, w, h, class (20 classes)
            target_list: [size(batch, 3, 13, 13, 6), size(batch, 3, 26, 26, 6), size(batch, 3, 52, 52, 6)] while 6 is [conf, x, y, w, h, class]
            anchors_list: [size(3, 2), size(3, 2), size(3, 2)]

        Returns:
            total loss of the boxes
        """
        total_loss = 0
        loss_distribution = {"no_obj": 0, "obj": 0, "box": 0, "cls": 0}
        loss_count = {"no_obj": 0, "obj": 0, "box": 0, "cls": 0}

        for predictions, target, anchors, weight in zip(predictions_list, target_list, anchors_list, self.balance_weight):
            if target.sum() == 0:
                continue
            # Check where obj and noobj (we ignore if target == -1)
            object_mask = target[..., 0] == 1
            not_object_mask = target[..., 0] == 0
            anchors = anchors.reshape(1, 3, 1, 1, 2)

            # compute no-object loss
            no_obj_loss = self.get_no_object_loss(predictions, target, not_object_mask)

            # compute object loss
            object_loss = self.get_object_loss(predictions, target, object_mask, anchors)

            # compute box coordinate loss
            box_loss = self.get_box_coordination_loss(predictions, target, object_mask, anchors)

            # compute classification loss
            cls_loss = self.get_class_prediction_loss(predictions, target, object_mask)

            # Update loss sums and counts
            loss_distribution["no_obj"] += no_obj_loss.item()
            loss_distribution["obj"] += object_loss.item()
            loss_distribution["box"] += box_loss.item()
            loss_distribution["cls"] += cls_loss.item()

            for key in loss_count:
                loss_count[key] += 1

            # compute total loss
            total_loss += weight * (self.lambda_box * box_loss + self.lambda_obj * object_loss + self.lambda_noobj * no_obj_loss + self.lambda_class * cls_loss)

        # Calculate average losses
        average_losses = {k: loss_distribution[k] / loss_count[k] for k in loss_distribution}

        return total_loss, average_losses
