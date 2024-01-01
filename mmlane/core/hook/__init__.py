# Copyright (c) OpenMMLab. All rights reserved.
from .ema import MEGVIIEMAHook
from .utils import is_parallel

__all__ = ['MEGVIIEMAHook', 'is_parallel']