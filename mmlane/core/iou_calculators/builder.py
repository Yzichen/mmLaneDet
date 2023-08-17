from mmcv.utils import Registry, build_from_cfg
from mmdet.core.bbox.iou_calculators.builder import IOU_CALCULATORS as MMDET_IOU_CALCULATORS

IOU_CALCULATORS = Registry('IoU calculator')


def build_iou_calculator(cfg, default_args=None):
    """Builder of IoU calculator."""
    if cfg['type'] in IOU_CALCULATORS._module_dict.keys():
        return build_from_cfg(cfg, IOU_CALCULATORS, default_args)
    else:
        return build_from_cfg(cfg, MMDET_IOU_CALCULATORS, default_args)