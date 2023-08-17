from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES
import warnings
from mmdet.models.losses import weight_reduce_loss


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    """
    :param feat: (B, C, H, W)
    :param ind: (B, K)
    :return:
        feat: (B, K, C)
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()    # (B, H, W, C)
    feat = feat.view(feat.size(0), -1, feat.size(3))    # (B, H*W, C)
    feat = _gather_feat(feat, ind)  # (B, K, C)
    return feat


def _reg_loss(regr, gt_regr, mask):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    L1 regression loss
    Args:
        regr:    (B, K, C)
        gt_regr: (B, K, C)
        mask:    (B, K)
    Returns:
    """
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()     # (B, K, C)
    isnotnan = (~torch.isnan(gt_regr)).float()
    mask *= isnotnan
    regr = regr * mask
    gt_regr = gt_regr * mask

    loss = torch.abs(regr - gt_regr)
    loss = loss.sum()
    # loss = loss / (num + 1e-4)
    loss = loss / torch.clamp_min(num, min=1.0)
    # import pdb; pdb.set_trace()
    return loss


@LOSSES.register_module()
class RegLossCenterNet(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """

    def __init__(self, loss_weight):
        super(RegLossCenterNet, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, output, target=None, ind=None, mask=None):
        """
        Args:
            output: (batch x dim x h x w) or (batch x max_objects)
            mask: (batch x max_objects)
            ind: (batch x max_objects)
            target: (batch x max_objects x dim)
        Returns:
        """
        if ind is None:
            pred = output
        else:
            pred = _transpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)
        return loss * self.loss_weight