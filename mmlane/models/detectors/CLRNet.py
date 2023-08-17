from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module
class CLRNet(SingleStageDetector):
    def __init__(self,
                 backbone,
                 aggregator=None,
                 neck=None,
                 lane_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(CLRNet, self).__init__(backbone=backbone, aggregator=aggregator, neck=neck, lane_head=lane_head,
                                     train_cfg=train_cfg, test_cfg=test_cfg, pretrained=pretrained,
                                     init_cfg=init_cfg)
