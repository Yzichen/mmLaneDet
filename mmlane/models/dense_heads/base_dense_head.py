# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch
from mmcv.cnn.utils.weight_init import constant_init
from mmcv.runner import BaseModule, force_fp32
import time

class BaseDenseHead(BaseModule, metaclass=ABCMeta):
    """Base class for DenseHeads."""

    def __init__(self, init_cfg=None):
        super(BaseDenseHead, self).__init__(init_cfg)

    def init_weights(self):
        super(BaseDenseHead, self).init_weights()
        # avoid init_cfg overwrite the initialization of `conv_offset`
        for m in self.modules():
            # DeformConv2dPack, ModulatedDeformConv2dPack
            if hasattr(m, 'conv_offset'):
                constant_init(m.conv_offset, 0)

    @abstractmethod
    def loss(self, **kwargs):
        """Compute losses of the head."""
        raise NotImplementedError

    @force_fp32(apply_to=('preds_dicts', ))
    def get_lanes(self,
                   preds_dicts,
                   img_metas=None,
                   with_nms=True,
                   **kwargs):
        """Transform network outputs of a batch into bbox results."""
        raise NotImplementedError

    def forward_train(self,
                      x,
                      img_metas,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_lanes (List):  List[(N_lanes0, 4+S), (N_lanes1, 4+S), ...]
            gt_labels (List): List[(N_lanes0, ), (N_lanes1, ), ...]
            gt_semantic_seg (Tensor): (B, H, W)
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        losses = self.loss(outs, img_metas, **kwargs)
        return losses

    def simple_test(self, feats, img_metas):
        """Test function without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n, ).
        """
        # torch.cuda.synchronize()
        # time1 = time.time()
        outs = self.forward(feats)
        # torch.cuda.synchronize()
        # time2 = time.time()
        # print("forward 外部 = ", (time2 - time1) * 1000)

        # torch.cuda.synchronize()
        # time1 = time.time()
        results_list = self.get_lanes(
            outs, img_metas=img_metas)
        # torch.cuda.synchronize()
        # time2 = time.time()
        # print("inf = ", (time2 - time1) * 1000)

        return results_list

    def simple_speed_test(self, *feats):
        outs = self.forward(*feats)
        return outs

