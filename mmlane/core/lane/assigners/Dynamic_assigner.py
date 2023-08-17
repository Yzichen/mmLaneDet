import torch
from .base_assigner import BaseAssigner
from ..builder import Lane_ASSIGNERS
from ..match_costs import build_match_cost
from ...iou_calculators import build_iou_calculator
from scipy.optimize import linear_sum_assignment


def distance_cost(predictions, targets, img_w):
    """
    repeat predictions and targets to generate all combinations
    use the abs distance as the new distance cost
    Args:
        predictions:  (num_priors, 4+S)
        targets: (num_targets, 4+S)
            4+S: 1 start_y (normalized), 1 start_x (absolute), 1 theta, 1 length, S coordinates(absolute)
    Returns:
        distances: (num_priors, num_targets)
    """
    num_priors = predictions.shape[0]
    num_targets = targets.shape[0]

    # (num_priors, S) --> (num_priors, 1, S) --> (num_priors, num_targets, S)
    predictions = predictions.unsqueeze(dim=1).repeat(1, num_targets, 1)[..., 4:]
    # (num_targets, S) --> (1, num_targets, S) --> (num_priors, num_targets, S)
    targets = targets.unsqueeze(dim=0).repeat(num_priors, 1, 1)[..., 4:]

    invalid_masks = (targets < 0) | (targets >= img_w)      # (num_priors, num_targets, S)
    lengths = (~invalid_masks).sum(dim=-1)   # (num_priors, num_targets)
    distances = torch.abs((targets - predictions))      # (num_priors, num_targets, S)
    distances[invalid_masks] = 0.
    distances = distances.sum(dim=-1) / (lengths.float() + 1e-9)    # (num_priors, num_targets)
    return distances


# For CLRNet
@Lane_ASSIGNERS.register_module
class DynamicAssigner(BaseAssigner):
    def __init__(self,
                 distance_cost_weight=3.0,
                 cls_cost=dict(type='FocalLossCost', weight=1.0),
                 iou_calculator=dict(type='LineIou', length=15),
                 dynamic_assign=True,
                 ):
        self.distance_cost_weight = distance_cost_weight
        self.cls_cost = build_match_cost(cls_cost)
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.dynamic_assign = dynamic_assign
        self.img_w = -1
        self.img_h = -1

    def dynamic_k_assign(self, cost, pair_wise_ious):
        """
        Assign grouth truths with priors dynamically.

        Args:
            cost: the assign cost. # (num_priors, num_target)
            pair_wise_ious: iou of grouth truth and priors.  # (num_priors, num_target)

        Returns:
            prior_idx: the index of assigned prior.     # (N_pos, )
            gt_idx: the corresponding ground truth index.   # (N_pos, )
        """
        matching_matrix = torch.zeros_like(cost)     # (num_priors, num_target)
        ious_matrix = pair_wise_ious
        ious_matrix[ious_matrix < 0] = 0.
        n_candidate_k = 4
        topk_ious, _ = torch.topk(ious_matrix, n_candidate_k, dim=0)  # (n_candidate_k=4, num_target)
        # 根据line iou计算动态的top_K的数量
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)  # (num_target, )
        num_gt = cost.shape[1]
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[:, gt_idx],  # (num_priors, )
                                    k=dynamic_ks[gt_idx].item(),
                                    largest=False)
            matching_matrix[pos_idx, gt_idx] = 1.0
        del topk_ious, dynamic_ks, pos_idx

        # 此时, 每个gt_lane对应dynamic_ks个priors，但可能存在一个prior对应多个gt_lanes.
        # 接下来, 强制每个prior只对应一个gt_lane, 这个gt_lane与prior的cost最小.
        matched_gt = matching_matrix.sum(1)  # (num_priors, )
        if (matched_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[matched_gt > 1, :], dim=1)
            # Todo 这里是不是应该把0改为：
            matching_matrix[matched_gt > 1, :] *= 0.0
            matching_matrix[matched_gt > 1, cost_argmin] = 1.0

        # (N_pos, ),  (N_pos, )
        prior_idx, gt_idx = matching_matrix.nonzero(as_tuple=True)
        return prior_idx, gt_idx

    def assign(self, cls_scores, lane_preds, gt_lanes, gt_labels):
        """
        :param cls_scores: (num_priors, n_cls)
        :param lane_preds: (num_priors, 4+S)
               4+S: 1 start_y(normalized), 1 start_x(normalized), 1 theta, 1 length (normalized), 72 coordinates(normalized),
        :param gt_lanes: (N_gt, 4+S)
               4+S: 1 start_y (normalized), 1 start_x (absolute), 1 theta (normalized), 1 length (absolute), S coordinates(absolute)
        :param gt_labels: (N_gt, )
        :return:
                matched_row_inds (Tensor): the index of assigned priors.     # (num_priors, )
                matched_col_inds (Tensor): the corresponding ground truth index.   # (num_priors, )
        """
        assert self.img_w > 0 and self.img_h > 0
        cls_scores = cls_scores.detach().clone()
        lane_preds = lane_preds.detach().clone()
        gt_lanes = gt_lanes.detach().clone()
        gt_labels = gt_labels.detach().clone()

        num_priors = lane_preds.shape[0]
        num_targets = gt_lanes.shape[0]

        # 1. assign -1 by default
        assigned_gt_inds = cls_scores.new_full((num_priors, ), -1, dtype=torch.long)
        assigned_labels = cls_scores.new_full((num_priors, ), -1, dtype=torch.long)

        if num_targets == 0 or num_priors == 0:
            # No ground truth or boxes, return empty assignment
            if num_targets == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return assigned_gt_inds, assigned_labels

        # 2. compute the weighted costs
        # 对齐 predictions 和 targets
        lane_preds[:, 1] *= (self.img_w - 1)  # normalized start_x --> absolute start_x
        lane_preds[:, 4:] *= (self.img_w - 1)  # normalized x_coods --> absolute x_coords

        # distances cost
        distances_score = distance_cost(lane_preds, gt_lanes, self.img_w)  # (num_priors, num_targets)
        distances_score = 1 - (distances_score / torch.max(distances_score)
                               ) + 1e-2  # normalize the distance

        # classification cost
        cls_cost = self.cls_cost(cls_scores, gt_labels.long())  # (num_priors, num_targets)

        # xy cost
        target_start_xys = gt_lanes[:, 0:2]  # (num_targets, 2)   2: 1 start_y (normalized), 1 start_x (absolute),
        target_start_xys[..., 0] *= (self.img_h - 1)  # normalized start_y --> absolute start_y
        prediction_start_xys = lane_preds[:, 0:2]  # (num_priors, 2)   2: 1 start_y (normalized), 1 start_x (absolute),
        prediction_start_xys[..., 0] *= (self.img_h - 1)  # normalized start_y --> absolute start_y
        # (num_priors, num_targets)
        start_xys_score = torch.cdist(prediction_start_xys, target_start_xys,
                                      p=2)
        start_xys_score = (1 - start_xys_score / torch.max(start_xys_score)) + 1e-2

        # theta cost
        target_thetas = gt_lanes[:, 2].unsqueeze(-1)
        # (num_priors, num_targets)
        theta_score = torch.cdist(lane_preds[:, 2].unsqueeze(-1),
                                  target_thetas,
                                  p=1).reshape(num_priors, num_targets) * 180
        theta_score = (1 - theta_score / torch.max(theta_score)) + 1e-2

        cost = -(distances_score * start_xys_score * theta_score
                 ) ** 2 * self.distance_cost_weight + cls_cost

        # 3. assignment
        # (N_pos, ),  (N_pos, )
        if self.dynamic_assign:
            iou = self.iou_calculator(lane_preds[..., 4:], gt_lanes[..., 4:], self.img_w, aligned=False)
            matched_row_inds, matched_col_inds = self.dynamic_k_assign(cost, iou)
        else:
            cost = cost.detach().cpu()
            matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
            matched_row_inds = torch.from_numpy(matched_row_inds).to(
                lane_preds.device)
            matched_col_inds = torch.from_numpy(matched_col_inds).to(
                lane_preds.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]

        return assigned_gt_inds, assigned_labels
