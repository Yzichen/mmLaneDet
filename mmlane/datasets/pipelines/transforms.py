import numpy as np
import mmcv
import random
import math
import imgaug.augmenters as iaa
from imgaug.augmentables.lines import LineString, LineStringsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import cv2
from ..builder import PIPELINES


@PIPELINES.register_module()
class FixedCrop:
    def __init__(self, crop):
        self.crop = crop    # (x_min, y_min, x_max, y_max)

    def _crop_data(self, results):
        img_shape = results['img_shape']    # (H, W, 3)
        x_min, y_min, x_max, y_max = self.crop

        for key in results.get('img_fields', ['img']):
            # crop the image
            img = results[key]
            img = img[y_min:y_max, x_min:x_max, ...]
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape
        results['crop'] = self.crop

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = results[key][y_min:y_max, x_min:x_max, ...]

        # update lanes
        for key in results.get('lane_fields', []):
            new_lanes = []
            for lane in results[key]:
                new_lane = []
                for p in lane:
                    new_lane.append((p[0]-x_min, p[1]-y_min))
                new_lanes.append(new_lane)
            results[key] = new_lanes

        return results

    def __call__(self, results):
        results = self._crop_data(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(crop={self.crop}, '
        return repr_str


@PIPELINES.register_module()
class RandomFlip:
    def __init__(self, flip_ratio=None, direction='horizontal'):
        if isinstance(flip_ratio, list):
            assert mmcv.is_list_of(flip_ratio, float)
            assert 0 <= sum(flip_ratio) <= 1
        elif isinstance(flip_ratio, float):
            assert 0 <= flip_ratio <= 1
        elif flip_ratio is None:
            pass
        else:
            raise ValueError('flip_ratios must be None, float, '
                             'or list of float')
        self.flip_ratio = flip_ratio

        valid_directions = ['horizontal', 'vertical', 'diagonal']
        if isinstance(direction, str):
            assert direction in valid_directions
        elif isinstance(direction, list):
            assert mmcv.is_list_of(direction, str)
            assert set(direction).issubset(set(valid_directions))
        else:
            raise ValueError('direction must be either str or list of str')
        self.direction = direction

        if isinstance(flip_ratio, list):
            assert len(self.flip_ratio) == len(self.direction)

    def lane_flip(self, lanes, img_shape, direction):
        """Flip bboxes horizontally.

        Args:
            lanes (list): [[(x_00,y0), (x_01,y1), ...], [(x_10,y0), (x_11,y1), ...], ...]
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction.
        Returns:
            new_lanes (list): [[(x_00,y0), (x_01,y1), ...], [(x_10,y0), (x_11,y1), ...], ...]
        """
        assert direction == 'horizontal'
        height, width, _ = img_shape
        if direction == 'horizontal':
            new_lanes = []
            for lane in lanes:
                new_lane = []
                for p in lane:
                    new_lane.append(((width - 1) - p[0], p[1]))
                new_lanes.append(new_lane)

        return new_lanes

    def flip_seg_label(self, seg_map):
        seg_map_copy = seg_map.copy()
        instances_ids = np.unique(seg_map[seg_map > 0])
        instances_ids = np.sort(instances_ids)
        instances_ids_reverse = np.flip(instances_ids)
        for i, instances_id in enumerate(instances_ids):
            seg_map[seg_map_copy == instances_id] = instances_ids_reverse[i]

        return seg_map

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added \
                into result dict.
        """
        # for vis
        # print("before transformation")
        # img_copy = results['img'].copy()
        # print(img_copy.shape)
        # print(img_copy.dtype)
        # lanes = results['gt_lanes']
        # seg_img = results['gt_semantic_seg']
        # for idx, lane in enumerate(lanes):
        #     for i in range(1, len(lane)):
        #         cv2.line(img_copy, tuple(map(int, lane[i-1])), tuple(map(int, lane[i])), (255, 0, 0), thickness=5)
        # cv2.imshow("line", img_copy)
        # cv2.imshow("raw img", results['img'])
        # cv2.imshow("seg img", seg_img)
        # print(results['lane_exist'])
        # cv2.waitKey(0)

        if 'flip' not in results:
            if isinstance(self.direction, list):
                # None means non-flip
                direction_list = self.direction + [None]
            else:
                # None means non-flip
                direction_list = [self.direction, None]

            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) -
                                                    1) + [non_flip_ratio]

            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)

            results['flip'] = cur_dir is not None
        if 'flip_direction' not in results:
            results['flip_direction'] = cur_dir
        if results['flip']:
            # flip image
            for key in results.get('img_fields', ['img']):
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction'])

            # flip bboxes
            for key in results.get('lane_fields', []):
                results[key] = self.lane_flip(results[key],
                                              results['img_shape'],
                                              results['flip_direction'])
            # flip segs
            for key in results.get('seg_fields', []):
                seg_map = mmcv.imflip(
                    results[key], direction=results['flip_direction'])
                # seg_classs_agnostic = results.get('seg_classs_agnostic', False)
                # if not seg_classs_agnostic:
                #     seg_map = self.flip_seg_label(seg_map)
                results[key] = seg_map

        # for vis
        # print("after transformation")
        # img_copy = results['img'].copy()
        # seg_img = results['gt_semantic_seg']
        # print(img_copy.shape)
        # print(img_copy.dtype)
        # lanes = results['gt_lanes']
        # for idx, lane in enumerate(lanes):
        #     for i in range(1, len(lane)):
        #         cv2.line(img_copy, tuple(map(int, lane[i-1])), tuple(map(int, lane[i])), (255, 0, 0), thickness=5)
        # cv2.imshow("line", img_copy)
        # cv2.imshow("raw img", results['img'])
        # cv2.imshow("seg img", seg_img)
        # cv2.waitKey(0)
        #
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(flip_ratio={self.flip_ratio})'


@PIPELINES.register_module
class RandomAffine:
    def __init__(self, affine_ratio, degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
        assert 0 <= affine_ratio <= 1
        self.affine_ratio = affine_ratio
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border

    def _transform_data(self, results, M, width, height):
        # transform img
        for key in results.get('img_fields', ['img']):
            img = results[key].copy()
            if (self.border[0] != 0) or (self.border[1] != 0) or (M != np.eye(3)).any():  # image changed
                if self.perspective:
                    im = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(0, 0, 0))
                else:  # affine
                    im = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(0, 0, 0))
            results[key] = im

        # transform lane
        for key in results.get('lane_fields', []):
            lanes = results[key].copy()
            new_lanes = []
            for lane in lanes:
                new_lane = []
                for p in lane:
                    p = np.expand_dims(np.array(p), axis=1)    # (2, 1)
                    p = np.concatenate((p, np.ones(shape=(1, 1), dtype=np.float)), axis=0)    # (3, 1)
                    new_p = (M[:2] @ p).squeeze().tolist()
                    new_lane.append(new_p)
                new_lanes.append(new_lane)
            results[key] = new_lanes

        # transform seg
        for key in results.get('seg_fields', []):
            seg = results[key].copy()
            if (self.border[0] != 0) or (self.border[1] != 0) or (M != np.eye(3)).any():  # image changed
                if self.perspective:
                    seg = cv2.warpPerspective(seg, M, dsize=(width, height), flags=cv2.INTER_NEAREST, borderValue=0)
                else:  # affine
                    seg = cv2.warpAffine(seg, M[:2], dsize=(width, height), flags=cv2.INTER_NEAREST, borderValue=0)
                results[key] = seg

    def __call__(self, results):
        # for vis
        # print("before transformation")
        # img_copy = results['img'].copy()
        # print(img_copy.shape)
        # print(img_copy.dtype)
        # lanes = results['gt_lanes']
        # for idx, lane in enumerate(lanes):
        #     for i in range(1, len(lane)):
        #         cv2.line(img_copy, tuple(map(int, lane[i-1])), tuple(map(int, lane[i])), (255, 0, 0), thickness=5)
        # cv2.imshow("line", img_copy)
        # cv2.imshow("raw img", results['img'])
        # cv2.imshow("seg img", results['gt_semantic_seg'])
        # cv2.waitKey(0)

        if random.random() < self.affine_ratio:
            img = results['img']
            height = img.shape[0] + self.border[0] * 2
            width = img.shape[1] + self.border[1] * 2

            # Center
            C = np.eye(3)
            C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
            C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

            # Perspective
            P = np.eye(3)
            P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
            P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

            # Rotation and Scale
            R = np.eye(3)
            a = random.uniform(-self.degrees, self.degrees)
            # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
            s = random.uniform(1 - self.scale, 1 + self.scale)
            # s = 2 ** random.uniform(-scale, scale)
            R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

            # Shear
            S = np.eye(3)
            S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
            S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

            # Translation
            T = np.eye(3)
            T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * width  # x translation (pixels)
            T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * height  # y translation (pixels)

            # Combined rotation matrix
            M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
            self._transform_data(results, M, width, height)

        # for vis
        # print("after transformation")
        # img_copy = results['img'].copy()
        # print(img_copy.shape)
        # print(img_copy.dtype)
        # lanes = results['gt_lanes']
        # for idx, lane in enumerate(lanes):
        #     for i in range(1, len(lane)):
        #         cv2.line(img_copy, tuple(map(int, lane[i-1])), tuple(map(int, lane[i])), (255, 0, 0), thickness=5)
        # cv2.imshow("line", img_copy)
        # cv2.imshow("raw img", results['img'])
        # cv2.imshow("seg img", results['gt_semantic_seg'])
        # cv2.waitKey(0)

        return results


@PIPELINES.register_module()
class Resize:
    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=False,
                 backend='cv2',
                 override=False):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.override = override

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``, \
                where ``img_scale`` is the selected image scale and \
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and upper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where \
                ``img_scale`` is sampled scale and None is just a placeholder \
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where \
                ``scale`` is sampled ratio multiplied with ``img_scale`` and \
                None is just a placeholder to be consistent with \
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into \
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get('img_fields', ['img']):
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results[key],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results[key].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results[key],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
            results[key] = img

            scale_factor = np.array([w_scale, h_scale], dtype=np.float32)
            results['img_shape'] = img.shape
            # in case that there is no padding
            results['pad_shape'] = img.shape
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio

    def _resize_lanes(self, results):
        """Resize lanes with ``results['scale_factor']``."""
        scale_factor = results['scale_factor']    # (w_scale, h_scale)
        for key in results.get('lane_fields', []):
            lanes = results[key]
            new_lanes = []
            for lane in lanes:
                new_lane = []
                for p in lane:
                    new_lane.append((p[0]*scale_factor[0], p[1]*scale_factor[1]))
                new_lanes.append(new_lane)
            results[key] = new_lanes

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results[key],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                gt_seg = mmcv.imresize(
                    results[key],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            results[key] = gt_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """
        # for vis
        # print("before transformation")
        # img_copy = results['img'].copy()
        # print(img_copy.shape)
        # lanes = results['gt_lanes']
        # for idx, lane in enumerate(lanes):
        #     for i in range(1, len(lane)):
        #         cv2.line(img_copy, tuple(map(int, lane[i-1])), tuple(map(int, lane[i])), (255, 0, 0), thickness=5)
        # cv2.imshow("line", img_copy)
        # cv2.imshow("raw img", results['img'])
        # cv2.imshow("seg img", results['gt_semantic_seg'])
        # cv2.waitKey(0)

        if 'scale' not in results:
            if 'scale_factor' in results:
                img_shape = results['img'].shape[:2]
                scale_factor = results['scale_factor']
                assert isinstance(scale_factor, float)
                results['scale'] = tuple(
                    [int(x * scale_factor) for x in img_shape][::-1])
            else:
                self._random_scale(results)
        else:
            if not self.override:
                assert 'scale_factor' not in results, (
                    'scale and scale_factor cannot be both set.')
            else:
                results.pop('scale')
                if 'scale_factor' in results:
                    results.pop('scale_factor')
                self._random_scale(results)

        self._resize_img(results)
        self._resize_lanes(results)
        self._resize_seg(results)

        # for vis
        # print("before transformation")
        # img_copy = results['img'].copy()
        # print(img_copy.shape)
        # lanes = results['gt_lanes']
        # for idx, lane in enumerate(lanes):
        #     for i in range(1, len(lane)):
        #         cv2.line(img_copy, tuple(map(int, lane[i-1])), tuple(map(int, lane[i])), (255, 0, 0), thickness=5)
        # cv2.imshow("line", img_copy)
        # cv2.imshow("raw img", results['img'])
        # cv2.imshow("seg img", results['gt_semantic_seg'])
        # cv2.waitKey(0)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'multiscale_mode={self.multiscale_mode}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        return repr_str


# @PIPELINES.register_module
# class ImgAug:
#     def __init__(self, transforms, with_lane=True, with_seg=False):
#         self.with_lane = with_lane
#         self.with_seg = with_seg
#
#         img_transforms = []
#         for aug in transforms:
#             p = aug['p']
#             if aug['name'] != 'OneOf':
#                 img_transforms.append(
#                     iaa.Sometimes(p=p,
#                                   then_list=getattr(
#                                       iaa,
#                                       aug['name'])(**aug['parameters'])))
#             else:
#                 img_transforms.append(
#                     iaa.Sometimes(
#                         p=p,
#                         then_list=iaa.OneOf([
#                             getattr(iaa,
#                                     aug_['name'])(**aug_['parameters'])
#                             for aug_ in aug['transforms']
#                         ])))
#
#         self.transform = iaa.Sequential(img_transforms)
#
#     def lane_to_linestrings(self, lanes):
#         lines = []
#         for lane in lanes:
#             lines.append(LineString(lane))
#
#         return lines
#
#     def linestrings_to_lanes(self, lines):
#         lanes = []
#         for line in lines:
#             lanes.append(line.coords)
#
#         return lanes
#
#     def _aug_data(self, results):
#         ori_img = results['img']
#         line_strings_ori = self.lane_to_linestrings(results['gt_lanes'])
#         line_strings_ori = LineStringsOnImage(line_strings_ori,
#                                               shape=ori_img.shape)
#         if self.with_seg:
#             mask_ori = SegmentationMapsOnImage(results['gt_semantic_seg'],
#                                                shape=ori_img.shape)
#             img, line_strings, seg = self.transform(
#                 image=ori_img.copy().astype(np.uint8),
#                 line_strings=line_strings_ori,
#                 segmentation_maps=mask_ori)
#         else:
#             img, line_strings = self.transform(
#                 image=ori_img.copy().astype(np.uint8),
#                 line_strings=line_strings_ori)
#
#         results['img'] = img
#         results['gt_lanes'] = self.linestrings_to_lanes(line_strings)
#         if self.with_seg:
#             results['gt_semantic_seg'] = mask_ori
#
#         return results
#
#     def __call__(self, results):
#         results = self._aug_data(results)
#         # for vis
#         # print("after transformation")
#         # img_copy = results['img'].copy()
#         # print(img_copy.shape)
#         # lanes = results['gt_lanes']
#         # for idx, lane in enumerate(lanes):
#         #     for i in range(1, len(lane)):
#         #         cv2.line(img_copy, lane[i-1].astype(np.int), lane[i].astype(np.int), (255, 0, 0), thickness=5)
#         # cv2.imshow("line", img_copy)
#         # cv2.imshow("raw img", results['img'])
#         # cv2.imshow("seg img", results['gt_semantic_seg'])
#         # cv2.waitKey(0)
#         return results


@PIPELINES.register_module()
class RandomUDOffsetLABEL:
    def __init__(self, ratio=0.5, max_offset=100):
        self.max_offset = max_offset
        self.ratio = ratio

    def _transform_data(self, results, offset):
        # transform img
        for key in results.get('img_fields', ['img']):
            img = results[key].copy()
            h, w = img.shape[:2]
            if offset > 0:
                img[offset:, :, :] = img[:h-offset, :, :]
                img[:offset, :, :] = 0
            if offset < 0:
                real_offset = -offset
                img[:h-real_offset, :, :] = img[real_offset:, :, :]
                img[h-real_offset:, :, :] = 0
            results[key] = img

        # transform seg
        for key in results.get('seg_fields', []):
            seg = results[key].copy()
            if offset > 0:
                seg[offset:, :] = seg[:h - offset, :]
                seg[:offset, :] = 0
            if offset < 0:
                real_offset = -offset
                seg[:h - real_offset, :] = seg[real_offset:, :]
                seg[h - real_offset:, :] = 0
            results[key] = seg

        # transform lane
        for key in results.get('lane_fields', []):
            lanes = results[key].copy()
            new_lanes = []
            for lane in lanes:
                new_lane = []
                for p in lane:  # p: [x, y]
                    p[1] += offset
                    new_lane.append(p)
                new_lanes.append(new_lane)
            results[key] = new_lanes

    def __call__(self, results):
        if random.random() < self.ratio:
            offset = np.random.randint(-self.max_offset, self.max_offset)
            self._transform_data(results, offset)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(ratio={self.ratio}, '
        repr_str += f'max_offset={self.max_offset}'
        return repr_str


@PIPELINES.register_module()
class RandomLROffsetLABEL:
    def __init__(self, ratio=0.5, max_offset=100):
        self.max_offset = max_offset
        self.ratio = ratio

    def _transform_data(self, results, offset):
        # transform img
        for key in results.get('img_fields', ['img']):
            img = results[key].copy()
            h, w = img.shape[:2]
            if offset > 0:
                img[:, offset:, :] = img[:, :w-offset, :]
                img[:, :offset, :] = 0
            if offset < 0:
                real_offset = -offset
                img[:, :w-real_offset, :] = img[:, real_offset:, :]
                img[:, w-real_offset:, :] = 0
            results[key] = img

        # transform seg
        for key in results.get('seg_fields', []):
            seg = results[key].copy()
            if offset > 0:
                seg[:, offset:] = seg[:, :w - offset]
                seg[:, :offset] = 0
            if offset < 0:
                real_offset = -offset
                seg[:, :w - real_offset] = seg[:, real_offset:]
                seg[:, w - real_offset:] = 0
            results[key] = seg

        # transform lane
        for key in results.get('lane_fields', []):
            lanes = results[key].copy()
            new_lanes = []
            for lane in lanes:
                new_lane = []
                for p in lane:  # p: [x, y]
                    p[0] += offset
                    new_lane.append(p)
                new_lanes.append(new_lane)
            results[key] = new_lanes

    def __call__(self, results):
        if random.random() < self.ratio:
            offset = np.random.randint(-self.max_offset, self.max_offset)
            self._transform_data(results, offset)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(ratio={self.ratio}, '
        repr_str += f'max_offset={self.max_offset}'
        return repr_str


@PIPELINES.register_module()
class RandomRotation(object):
    def __init__(self, rotate_ratio=0.5, degree=(-10, 10)):
        self.rotate_ratio = rotate_ratio
        self.degree = degree

    def _transform_data(self, results, rotate_matrix):
        # transform img
        for key in results.get('img_fields', ['img']):
            img = results[key].copy()
            height, width = img.shape[0:2]
            img = cv2.warpAffine(img, rotate_matrix[:2], dsize=(width, height), borderValue=(0, 0, 0))
            results[key] = img

        # transform seg
        for key in results.get('seg_fields', []):
            seg = results[key].copy()
            seg = cv2.warpAffine(seg, rotate_matrix[:2], dsize=(width, height), flags=cv2.INTER_NEAREST,
                                 borderValue=(0, 0, 0))
            results[key] = seg

        # transform lane
        for key in results.get('lane_fields', []):
            lanes = results[key].copy()
            new_lanes = []
            for lane in lanes:
                new_lane = []
                for p in lane:
                    p = np.expand_dims(np.array(p), axis=1)    # (2, 1)
                    p = np.concatenate((p, np.ones(shape=(1, 1), dtype=np.float)), axis=0)    # (3, 1)
                    new_p = (rotate_matrix[:2] @ p).squeeze().tolist()
                    new_lane.append(new_p)
                new_lanes.append(new_lane)
            results[key] = new_lanes

    def __call__(self, results):
        if random.random() < self.rotate_ratio:
            degree = random.uniform(self.degree[0], self.degree[1])
            h, w = results['img'].shape[0:2]
            center = (w / 2, h / 2)
            rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=degree, scale=1.0)
            self._transform_data(results, rotate_matrix)

        return results

