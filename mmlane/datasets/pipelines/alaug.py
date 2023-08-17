import random
import collections
import albumentations as al
import numpy as np
from ..builder import PIPELINES


@PIPELINES.register_module
class Alaug(object):
    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        # init as None
        self.__augmentor = None
        # put transforms in a list
        self.transforms = []
        self.bbox_params = None
        self.keypoint_params = None

        for transform in transforms:
            if isinstance(transform, dict):
                if transform['type'] == 'Compose':
                    self.get_al_params(transform['params'])
                else:
                    transform = self.build_transforms(transform)
                    if transform is not None:
                        self.transforms.append(transform)
            else:
                raise TypeError('transform must be a dict')
        self.build()

    def get_al_params(self, compose):
        if compose['bboxes']:
            self.bbox_params = al.BboxParams(
                format='pascal_voc',
                min_area=0.0,
                min_visibility=0.0,
                label_fields=["bbox_labels"])
        if compose['keypoints']:
            self.keypoint_params = al.KeypointParams(
                format='xy', remove_invisible=False)

    def build_transforms(self, transform):
        if transform['type'] == 'OneOf':
            transforms = transform['transforms']
            choices = []
            for t in transforms:
                parmas = {
                    key: value
                    for key, value in t.items() if key != 'type'
                }
                choice = getattr(al, t['type'])(**parmas)
                choices.append(choice)
            return getattr(al, 'OneOf')(transforms=choices, p=transform['p'])

        parmas = {
            key: value
            for key, value in transform.items() if key != 'type'
        }
        return getattr(al, transform['type'])(**parmas)

    def build(self):
        if len(self.transforms) == 0:
            return
        self.__augmentor = al.Compose(
            self.transforms,
            bbox_params=self.bbox_params,
            keypoint_params=self.keypoint_params,
        )

    def cal_sum_list(self, itmes, index):
        sum = 0
        for i in range(index):
            sum += itmes[i]
        return sum

    def __call__(self, results):
        if self.__augmentor is None:
            return results
        img = results['img']
        if 'gt_bboxes' in results:
            gt_bboxes = results['gt_bboxes']
            bboxes = []
            bbox_labels = []
            for i in range(np.shape(gt_bboxes)[0]):
                if (gt_bboxes[i, 0] == gt_bboxes[i, 2]) | (
                        gt_bboxes[i, 1] == gt_bboxes[i, 3]):
                    pass
                else:
                    b = gt_bboxes[i, :]
                    b = np.concatenate((b, [i]))
                    bboxes.append(b)
                    bbox_labels.append(results['gt_labels'][i])
        else:
            bboxes = None
            bbox_labels = None

        if 'gt_semantic_seg' in results:
            masks = results['gt_semantic_seg']
        else:
            masks = None

        if 'gt_lanes' in results:
            lane_points = results["gt_lanes"]
            lane_points_nums = []
            for pts in lane_points:
                lane_points_nums.append(len(pts))

            points_val = []
            for pts in lane_points:
                num = len(pts)
                for i in range(num):
                    points_val.append(list(pts[i]))

        aug = self.__augmentor(
            image=img,
            keypoints=points_val,
            bboxes=bboxes,
            mask=masks,
            bbox_labels=bbox_labels)

        results['img'] = aug['image']
        results['img_shape'] = results['img'].shape
        if 'gt_bboxes' in results:
            if aug['bboxes']:
                results['gt_bboxes'] = np.array(aug['bboxes'])[:, :4]
                results['gt_labels'] = np.array(aug['bbox_labels'])
            else:
                return None

        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = np.array(aug['mask'])

        if 'gt_lanes' in results:
            points = aug['keypoints']
            lane_points_list = [[] for i in range(len(lane_points_nums))]
            for lane_id in range(len(lane_points_nums)):
                for i in range(lane_points_nums[lane_id]):
                    lane_points_list[lane_id].append(points[self.cal_sum_list(lane_points_nums, lane_id) + i])
            results['gt_lanes'] = lane_points_list

        # for vis
        # import cv2
        # print("after transformation")
        # img_copy = results['img'].copy()
        # seg_img = results['gt_semantic_seg']
        # print(img_copy.shape)
        # print(img_copy.dtype)
        # lanes = results['gt_lanes']
        # for idx, lane in enumerate(lanes):
        #     for i in range(1, len(lane)):
        #         cv2.line(img_copy, tuple(map(int, lane[i-1])), tuple(map(int, lane[i])), (255, 0, 0), thickness=5)
        # cv2.imwrite("line.png", img_copy)
        # cv2.imwrite("raw_img.png", results['img'])
        # cv2.imwrite("seg_img.png", seg_img)
        # cv2.waitKey(0)

        return results

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
