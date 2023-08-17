import torch
from .builder import IOU_CALCULATORS


@IOU_CALCULATORS.register_module
class LineIou:
    def __init__(self, length=15):
        self.length = length

    def __call__(self, line_pred, line_target, img_w, aligned=True):
        """
        :param line_pred: (N_pred, S)
        :param line_target: (N_gt, S)
        :return:
            iou: (num_pred, N_gt)
        """
        px1 = line_pred - self.length
        px2 = line_pred + self.length
        tx1 = line_target - self.length
        tx2 = line_target + self.length

        if aligned:
            invalid_mask = line_target  # (N_pos, S)
            ovr = torch.min(px2, tx2) - torch.max(px1, tx1)  # (N_pos, S)
            union = torch.max(px2, tx2) - torch.min(px1, tx1)  # (N_pos, S)
        else:
            num_pred = line_pred.shape[0]
            invalid_mask = line_target.repeat(num_pred, 1, 1)    # (num_pred, num_target, S)
            # (num_pred, num_target, S)
            ovr = (torch.min(px2[:, None, :], tx2[None, ...]) -
                   torch.max(px1[:, None, :], tx1[None, ...]))      # (num_pred, num_target, S)
            union = (torch.max(px2[:, None, :], tx2[None, ...]) -
                     torch.min(px1[:, None, :], tx1[None, ...]))    # (num_pred, num_target, S)

        invalid_masks = (invalid_mask < 0) | (invalid_mask >= img_w)  # (num_pred, num_target, S)
        ovr[invalid_masks] = 0.
        union[invalid_masks] = 0.
        iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)  # (num_pred, num_target)
        return iou


