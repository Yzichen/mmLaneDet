import torch.nn as nn
from mmcv.utils import Registry, build_from_cfg
from mmdet.models.utils import Transformer as MMDET_Transformer

TRANSFORMER = Registry('Transformer')


def build_transformer(cfg, default_args=None):
    """Builder for Transformer."""
    if cfg['type'] in TRANSFORMER._module_dict.keys():
        return build_from_cfg(cfg, TRANSFORMER, default_args)
    else:
        return MMDET_Transformer.build(cfg)