import torch
from scipy.optimize import linear_sum_assignment
from ..builder import Lane_ASSIGNERS
from ..match_costs import build_match_cost
from .base_assigner import BaseAssigner
from ..structures import BezierCurve
from mmdet.core import AssignResult


@Lane_ASSIGNERS.register_module()
class BezierHungarianAssigner(BaseAssigner):
    def __init__(self, order=3, num_sample_points=100, alpha=0.8, window_size=0):
        self.num_sample_points = num_sample_points
        self.alpha = alpha
        self.bezier_curve = BezierCurve(order=order)
        self.window_size = window_size

    def assign(self, cls_logits, pred_control_points, gt_control_points, gt_labels):
        """
        :param cls_logits: (N_q, n_cls=1)
        :param pred_control_points: (N_q, 4, 2)
        :param gt_control_points: (N_gt, 4, 2)
        :param gt_labels: (N_gt, )
        :return:
            assigned_gt_inds: (N_q, )
            assigned_labels:  (N_q, )
        """
        N_q = cls_logits.shape[0]
        N_gt = gt_labels.shape[0]

        # 1. assign -1 by default
        assigned_gt_inds = cls_logits.new_full((N_q, ), -1, dtype=torch.long)     # (N_q, )
        assigned_labels = cls_logits.new_full((N_q, ), -1, dtype=torch.long)      # (N_q, )

        if N_q == 0 or N_gt == 0:
            # No ground truth or boxes, return empty assignment
            if N_gt == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return assigned_gt_inds, assigned_labels

        # 2. compute the costs
        # cls_cost
        cls_score = cls_logits.sigmoid()    # (N_q, n_cls=1)

        # Local maxima prior
        if self.window_size > 0:
            _, max_indices = torch.nn.functional.max_pool1d(cls_score.unsqueeze(0).permute(0, 2, 1),
                                                            kernel_size=self.window_size, stride=1,
                                                            padding=(self.window_size - 1) // 2, return_indices=True)
            max_indices = max_indices.squeeze()   # (N_q, )
            indices = torch.arange(0, N_q, dtype=cls_score.dtype, device=cls_score.device)
            local_maxima = (max_indices == indices)     # (N_q)
        else:
            local_maxima = cls_score.new_ones((N_q, ))

        cls_score = cls_score.squeeze(dim=1)     # (N_q, )
        cls_cost = (local_maxima * cls_score).unsqueeze(dim=1).repeat(1, N_gt)   # (N_q, N_gt)

        # curve sampling cost
        # (N_q, N_sample_points, 2)
        pred_sample_points = self.bezier_curve.get_sample_points(control_points_matrix=pred_control_points,
                                                                 num_sample_points=100)
        # (N_gt, N_sample_points, 2)
        gt_sample_points = self.bezier_curve.get_sample_points(control_points_matrix=gt_control_points,
                                                               num_sample_points=100)

        # (N_q, N_sample_points, 2) --> (N_q, N_sample_points*2)
        pred_sample_points = pred_sample_points.flatten(start_dim=-2)
        # (N_gt, N_sample_points, 2) --> (N_gt, N_sample_points*2)
        gt_sample_points = gt_sample_points.flatten(start_dim=-2)
        reg_cost = 1 - torch.cdist(pred_sample_points, gt_sample_points, p=1) / self.num_sample_points  # (Nq, N_gt)
        reg_cost = reg_cost.clamp(min=0, max=1)

        cost = -cls_cost ** (1 - self.alpha) * reg_cost ** self.alpha

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        cost = torch.nan_to_num(cost, nan=100.0, posinf=100.0, neginf=-100.0)
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(cls_score.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(cls_score.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]

        return assigned_gt_inds, assigned_labels
