# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import Registry, build_from_cfg
from mmdet.core.bbox.match_costs.builder import MATCH_COST as MMDET_MATCH_COST

MATCH_COST = Registry('Match Cost')


def build_match_cost(cfg, default_args=None):
    """Builder of IoU calculator."""
    if cfg['type'] in MATCH_COST._module_dict.keys():
        return build_from_cfg(cfg, MATCH_COST, default_args)
    else:
        return build_from_cfg(cfg, MMDET_MATCH_COST, default_args)

