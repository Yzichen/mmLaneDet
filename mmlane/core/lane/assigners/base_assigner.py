from abc import ABCMeta, abstractmethod


class BaseAssigner(metaclass=ABCMeta):
    """Base assigner that assigns lanes to ground truth lanes."""

    @abstractmethod
    def assign(self, cls_scores, lane_preds, gt_lanes):
        """Assign lanes to either a ground truth lanes or a negative lanes."""