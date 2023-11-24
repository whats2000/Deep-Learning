import torch
import torch.nn as nn
import torch.nn.functional as F


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


class YoloLossV2(YoloLoss):
    def __init__(self, s, b, l_coord, l_noobj):
        """
        Yolo Loss V2 with Focal Loss and Varifocal Loss.

        Args:
            S: Grid size
            B: Number of bounding boxes
            l_coord: Coefficient for the coordinate loss
            l_noobj: Coefficient for the no-object loss
        """
        super(YoloLossV2, self).__init__(s, b, l_coord, l_noobj)

    @staticmethod
    def varifocal_loss(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """
        Computes Varifocal Loss.

        Args:
            pred_score: Predicted scores
            gt_score: Ground truth scores
            label: Labels
            alpha: Alpha value for balancing
            gamma: Gamma value for focusing

        Returns:
            Varifocal loss value
        """
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction='none') *
                    weight).mean(1).sum()
        return loss

    @staticmethod
    def focal_loss(pred, label, gamma=1.5, alpha=0.25):
        """
        Computes Focal Loss.

        Args:
            pred: Predicted logits
            label: Ground truth labels
            gamma: Gamma value for focusing
            alpha: Alpha value for balancing

        Returns:
            Focal loss value
        """
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
        pred_prob = pred.sigmoid()
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()

    def get_class_prediction_loss(self, classes_pred, classes_target, has_object_map):
        """
        Calculate the class prediction loss using Varifocal Loss.

        Args:
            classes_pred: tensor of size (batch_size, S, S, 20), predicted class probabilities.
            classes_target: tensor of size (batch_size, S, S, 20), ground truth for class presence.
            has_object_map: tensor of size (batch_size, S, S), indicating which cells contain objects.

        Returns:
            class_loss: scalar, the total class prediction loss calculated using Varifocal Loss.

        This method calculates the loss for the class predictions in cells that contain objects.
        It uses Varifocal Loss to focus more on misclassified examples and address class imbalance.
        """
        object_mask = has_object_map.unsqueeze(-1).expand_as(classes_pred)
        classes_pred = classes_pred[object_mask].view(-1, 20)  # Predicted class scores
        classes_target = classes_target[object_mask].view(-1, 20)  # Ground truth class scores

        # Ground truth scores for object classes (assuming binary classification for each class)
        gt_scores = torch.ones_like(classes_target)

        # Apply Varifocal Loss
        class_loss = self.varifocal_loss(classes_pred, gt_scores, classes_target).sum()

        return class_loss

    def get_no_object_loss(self, pred_boxes_list, has_object_map):
        """
        Calculate the no-object loss using Varifocal Loss.

        Args:
            pred_boxes_list: list of tensors, each tensor is of size (Batch, S, S, 5), representing B predicted bounding boxes.
            has_object_map: tensor of size (Batch, S, S), a boolean mask indicating cells containing objects.

        Returns:
            no_obj_loss: scalar, the computed no-object loss.

        This function computes the Varifocal Loss for the cells that do not contain any objects. The loss is calculated
        on the confidence score predictions of these cells, comparing them against a ground truth score of 0.
        """
        not_object_mask = ~has_object_map

        repeated_not_object_mask = not_object_mask.unsqueeze(-1).repeat(1, 1, 1, self.B).reshape(-1)
        flat_pred_conf = torch.cat([pred_boxes[..., 4].reshape(-1) for pred_boxes in pred_boxes_list], dim=0)
        pred_conf_no_obj = flat_pred_conf[repeated_not_object_mask]

        # Ground truth scores for non-object cells are 0
        gt_conf_no_obj = torch.zeros_like(pred_conf_no_obj)

        # Applying Varifocal Loss
        no_obj_loss = self.varifocal_loss(pred_conf_no_obj, gt_conf_no_obj, torch.zeros_like(gt_conf_no_obj))

        return no_obj_loss

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
