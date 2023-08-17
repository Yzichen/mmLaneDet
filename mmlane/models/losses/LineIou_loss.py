import torch
import torch.nn as nn

from ..builder import LOSSES
from mmdet.models.losses import weight_reduce_loss


def line_iou_loss(preds, targets, length, img_w, weight=None, reduction='mean', avg_factor=None):
    px1 = preds - length
    px2 = preds + length
    tx1 = targets - length
    tx2 = targets + length

    invalid_mask = targets  # (N, S)
    ovr = torch.min(px2, tx2) - torch.max(px1, tx1)  # (N, S)
    union = torch.max(px2, tx2) - torch.min(px1, tx1)  # (N, S)

    invalid_masks = (invalid_mask < 0) | (invalid_mask >= img_w)  # (N, S)
    ovr[invalid_masks] = 0.
    union[invalid_masks] = 0.
    iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)  # (N, )
    iou_loss = 1 - iou  # (N, )
    iou_loss = weight_reduce_loss(
        iou_loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return iou_loss


@LOSSES.register_module()
class LineIou_Loss(nn.Module):
    def __init__(self, length=15, reduction='mean', loss_weight=1.0):
        super(LineIou_Loss, self).__init__()
        self.length = length
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self,
                preds,
                targets,
                img_w,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ):
        """
        :param preds: (N, S)  S: absolute x coords
        :param targets: (N, S)  S: absolute x coords
        :param img_w: image width
        :param weigth: (N, )
        :param avg_factor: int
        :return:
            iou_loss: float
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss_iou = self.loss_weight * line_iou_loss(
            preds=preds,
            targets=targets,
            length=self.length,
            img_w=img_w,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor
        )
        return loss_iou
