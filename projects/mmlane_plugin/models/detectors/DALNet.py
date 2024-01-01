from mmlane.models.builder import DETECTORS
from mmlane.models.detectors.single_stage import SingleStageDetector
import time
import torch

@DETECTORS.register_module
class DALNet(SingleStageDetector):
    def __init__(self,
                 backbone,
                 aggregator=None,
                 neck=None,
                 lane_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(DALNet, self).__init__(backbone=backbone, aggregator=aggregator, neck=neck, lane_head=lane_head,
                                     train_cfg=train_cfg, test_cfg=test_cfg, pretrained=pretrained,
                                     init_cfg=init_cfg)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        # torch.cuda.synchronize()
        # time1 = time.time()
        x = self.backbone(img)
        # torch.cuda.synchronize()
        # time2 = time.time()
        # print("backbone = ", (time2 - time1) * 1000)

        x = list(x)
        if self.with_aggregator:
            # torch.cuda.synchronize()
            # time1 = time.time()
            x[-1] = self.aggregator(x[-1])
            # torch.cuda.synchronize()
            # time2 = time.time()
            # print("aggregator = ", (time2 - time1) * 1000)

        # torch.cuda.synchronize()
        # time1 = time.time()
        if self.with_neck:
            x = self.neck(x)
        # torch.cuda.synchronize()
        # time2 = time.time()
        # print("neck = ", (time2 - time1) * 1000)
        return x