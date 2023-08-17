import torch
from .builder import MATCH_COST


@MATCH_COST.register_module()
class LaneL1Cost:
    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, lanes_pred, gt_lanes):
        """
        :param lanes_pred: (N_pred, S)  normalized
        :param gt_lanes: (N_gt, S)      normalized
        :return:
            cost: (N_pred, N_gt)
        """
        num_preds = lanes_pred.shape[0]
        num_targets = gt_lanes.shape[0]

        lanes_pred = lanes_pred.unsqueeze(dim=1).repeat(1, num_targets, 1)  # (num_preds, num_targets, S)
        gt_lanes = gt_lanes.unsqueeze(dim=0).repeat(num_preds, 1, 1)    # (num_preds, num_targets, S)

        invalid_masks = (gt_lanes < 0) | (gt_lanes >= 1)  # (num_preds, num_targets, S)
        lengths = (~invalid_masks).sum(dim=-1)  # (num_preds, num_targets)
        distances = torch.abs((gt_lanes - lanes_pred))  # (num_preds, num_targets, S)
        distances[invalid_masks] = 0.
        lane_cost = distances.sum(dim=-1) / (lengths.float() + 1e-9)  # (num_priors, num_targets)

        return lane_cost * self.weight


@MATCH_COST.register_module()
class LaneIouCost:
    def __init__(self, weight=1.0, length=15):
        self.weight = weight
        self.length = length

    def __call__(self, lanes_pred, gt_lanes, img_w):
        """
        :param lanes_pred: (N_pred, S)  normalized
        :param gt_lanes: (N_gt, S)      normalized
        :return:
            cost: (N_pred, N_gt)
        """
        px1 = lanes_pred - self.length
        px2 = lanes_pred + self.length
        tx1 = gt_lanes - self.length
        tx2 = gt_lanes + self.length

        num_pred = lanes_pred.shape[0]
        invalid_mask = gt_lanes.repeat(num_pred, 1, 1)    # (num_pred, num_target, S)
        # (num_pred, num_target, S)
        ovr = (torch.min(px2[:, None, :], tx2[None, ...]) -
               torch.max(px1[:, None, :], tx1[None, ...]))      # (num_pred, num_target, S)
        union = (torch.max(px2[:, None, :], tx2[None, ...]) -
                 torch.min(px1[:, None, :], tx1[None, ...]))    # (num_pred, num_target, S)

        invalid_masks = (invalid_mask < 0) | (invalid_mask >= img_w)  # (num_pred, num_target, S)
        ovr[invalid_masks] = 0.
        union[invalid_masks] = 0.
        iou_cost = 1 - ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)  # (N_pred, num_target)

        return iou_cost * self.weight


@MATCH_COST.register_module()
class RegCost:
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, pred, gt):
        reg_cost_yl = torch.cdist(pred, gt, p=1) * self.weight
        return reg_cost_yl
