from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module
class CondLane(SingleStageDetector):
    def __init__(self,
                 backbone,
                 aggregator=None,
                 neck=None,
                 lane_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(CondLane, self).__init__(backbone=backbone, aggregator=aggregator, neck=neck, lane_head=lane_head,
                                       train_cfg=train_cfg, test_cfg=test_cfg, pretrained=pretrained,
                                       init_cfg=init_cfg)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        x = list(x)
        if self.with_aggregator:
            x[-1] = self.aggregator(x[-1])
        if self.with_neck:
            x = self.neck(x)

        return x
