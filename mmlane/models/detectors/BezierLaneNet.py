from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from ..utils import DilatedBlocks

@DETECTORS.register_module
class BezierLaneNet(SingleStageDetector):
    def __init__(self,
                 backbone,
                 dilated_blocks=None,
                 aggregator=None,
                 neck=None,
                 lane_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(BezierLaneNet, self).__init__(backbone=backbone, aggregator=aggregator, neck=neck, lane_head=lane_head,
                                            train_cfg=train_cfg, test_cfg=test_cfg, pretrained=pretrained,
                                            init_cfg=init_cfg)
        if dilated_blocks is not None:
            self.dilated_blocks = DilatedBlocks(**dilated_blocks)
        else:
            self.dilated_blocks = None

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        f = x[0]
        if self.dilated_blocks is not None:
            x = self.dilated_blocks(f)

        if self.with_neck:
            x = self.neck(x)
        if self.with_aggregator:
            x = self.aggregator(x)

        return x, f

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x, f = self.extract_feat(img)
        outs = self.lane_head(x, f)
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
        x, f = self.extract_feat(img)
        losses = self.lane_head.forward_train(x, f, img_metas, **kwargs)
        return losses

    def simple_test(self, img, img_metas, **kwargs):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x, f = self.extract_feat(img)
        results_list = self.lane_head.simple_test(
            x, img_metas)

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
        x, f = self.extract_feat(img)
        results_list = self.lane_head.simple_speed_test(x, f)

        return results_list