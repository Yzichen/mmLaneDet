import torch
import warnings
from ..builder import DETECTORS, build_backbone, build_head, build_neck, build_aggregator
from .base import BaseDetector
import time

@DETECTORS.register_module
class SingleStageDetector(BaseDetector):
    def __init__(self,
                 backbone,
                 aggregator=None,
                 neck=None,
                 lane_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SingleStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if aggregator is not None:
            self.aggregator = build_aggregator(aggregator)
        if neck is not None:
            self.neck = build_neck(neck)
        lane_head.update(train_cfg=train_cfg)
        lane_head.update(test_cfg=test_cfg)
        self.lane_head = build_head(lane_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        # torch.cuda.synchronize()
        # time1 = time.time()
        x = self.backbone(img)
        # torch.cuda.synchronize()
        # time2 = time.time()
        # print("backbone = ", (time2 - time1) * 1000)

        # torch.cuda.synchronize()
        # time1 = time.time()
        if self.with_aggregator:
            x = self.aggregator(x)
        # torch.cuda.synchronize()
        # time2 = time.time()
        # print("transform = ", (time2 - time1) * 1000)

        # torch.cuda.synchronize()
        # time1 = time.time()
        if self.with_neck:
            x = self.neck(x)
        # torch.cuda.synchronize()
        # time2 = time.time()
        # print("neck = ", (time2 - time1) * 1000)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.lane_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      **kwargs
                      ):
        """
        Args:
            img (Tensor): Input images of shape (B, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_lanes (List):  List[(N_lanes0, 4+S), (N_lanes1, 4+S), ...]
            gt_labels (List): List[(N_lanes0, ), (N_lanes1, ), ...]
            gt_semantic_seg (Tensor): (B, H, W)

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.lane_head.forward_train(x, img_metas, **kwargs)
        return losses

    def simple_test(self, img, img_metas=None, **kwargs):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        results_list = self.lane_head.simple_test(
            feat, img_metas)

        return results_list

    def simple_speed_test(self, img):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        results_list = self.lane_head.simple_speed_test(feat)

        return results_list

