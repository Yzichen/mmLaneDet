import mmcv
from ..builder import PIPELINES
import os.path as osp
import numpy as np


@PIPELINES.register_module()
class LoadImageFromFile:
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 channel_order='bgr',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, channel_order=self.channel_order)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']

        # for vis
        # import cv2
        # img = results['img']
        # cv2.imwrite("img.png", img)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"channel_order='{self.channel_order}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module
class LoadLaneAnnotations:
    def __init__(self, with_lane=True, with_seg=False, with_lane_exist=False,
                 file_client_args=dict(backend='disk'), seg_classs_agnostic=False,
                 cls_agnostic=True):
        self.with_lane = with_lane
        self.with_seg = with_seg
        self.with_lane_exist = with_lane_exist
        self.seg_classs_agnostic = seg_classs_agnostic
        self.cls_agnostic = cls_agnostic
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_lanes(self, results):
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """

        ann_info = results['ann_info']
        results['gt_lanes'] = ann_info['lanes'].copy()
        results['gt_labels'] = ann_info['lanes_labels'].copy()
        if self.cls_agnostic:
            results['gt_labels'] *= 0
        results['lane_fields'].append('gt_lanes')
        return results

    def _load_semantic_seg(self, results):
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = osp.join(results['seg_prefix'], results['ann_info']['seg_map'])
        img_bytes = self.file_client.get(filename)
        seg_map = mmcv.imfrombytes(
            img_bytes, flag='unchanged').squeeze()
        if len(seg_map.shape) > 2:
            seg_map = seg_map[:, :, 0]
        if self.seg_classs_agnostic:
            seg_map[seg_map > 0] = 1
        seg_map = seg_map.squeeze()
        results['gt_semantic_seg'] = seg_map
        results['seg_fields'].append('gt_semantic_seg')
        results['seg_classs_agnostic'] = self.seg_classs_agnostic
        return results

    def _load_lane_exist(self, results):
        ann_info = results['ann_info']
        results['lane_exist'] = ann_info['lane_exist'].copy()
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        if self.with_lane:
            results = self._load_lanes(results)
            if results is None:
                return None
        if self.with_seg and results['ann_info'].get('seg_map', None) is not None:
            results = self._load_semantic_seg(results)

        if self.with_lane_exist and results['ann_info'].get('lane_exist', None) is not None:
            results = self._load_lane_exist(results)

        # for vis
        # import cv2
        # img = results['img']
        # gt_lanes = results['gt_lanes']
        # for lane in gt_lanes:
        #     for i in range(len(lane) - 1):
        #         point0 = (int(lane[i][0]), int(lane[i][1]))
        #         point1 = (int(lane[i+1][0]), int(lane[i+1][1]))
        #         cv2.line(img, point0, point1, color=(255, 0, 0), thickness=5)
        # cv2.imwrite("img.png", img)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_lane={self.with_lane}, '
        repr_str += f'with_seg={self.with_seg}, '
        return repr_str