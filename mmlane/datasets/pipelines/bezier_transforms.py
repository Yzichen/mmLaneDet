import cv2
import torch
import mmcv
import numpy as np
from ..builder import PIPELINES
from scipy.special import comb as n_over_k
from mmlane.core import BezierCurve
from scipy.interpolate import splprep, splev
from mmcv.parallel import DataContainer as DC
from .formating import to_tensor


@PIPELINES.register_module()
class Lanes2ControlPoints:
    def __init__(self, order=3, interpolate=False, fix_endpoints=False):
        self.num_points = order + 1
        self.interpolate = interpolate
        self.fix_endpoints = fix_endpoints
        self.bezier_curve = BezierCurve(order=order)

    def get_bezier_coefficient(self):
        # n: 表示控制点的数量;  t: [0, 1]的采样点, k: 表示控制点的索引(在公式中为i=0,1, ..., n)
        # 系数Bkn(t) = t**k * (1-t)**(n-k) * Cni;    Cni = n!/(k!(n-k)!)
        Mtk = lambda n, t, k: t ** k * (1 - t) ** (n - k) * n_over_k(n, k)
        BezierCoeff = lambda ts: [[Mtk(self.num_points - 1, t, k) for k in range(self.num_points)] for t in ts]

        return BezierCoeff

    def interpolate_lane(self, lanes, n=100):
        # Spline interpolation of a lane. Used on the predictions
        x, y = lanes[:, 0], lanes[:, 1]
        tck, _ = splprep([x, y], s=0, t=n, k=min(3, len(x) - 1))

        u = np.linspace(0., 1., n)
        return np.stack(splev(u, tck), axis=-1)

    def __call__(self, results):
        gt_lanes = results['gt_lanes']
        control_points_list = []
        interpolate_lanes = []
        for lanes in gt_lanes:
            lanes = np.array(lanes, dtype=np.float32)
            if self.interpolate:
                lanes = self.interpolate_lane(lanes)
            interpolate_lanes.append(lanes)
            if self.fix_endpoints:
                control_points = self.bezier_curve.get_control_points_with_fixed_endpoints(lanes, to_list=True)
            else:
                control_points = self.bezier_curve.get_control_points(lanes, to_list=True)
            control_points_list.append(control_points)

        results['control_points'] = control_points_list
        lane_fields = results['lane_fields']
        gt_lanes_index = lane_fields.index('gt_lanes')
        results['lane_fields'].pop(gt_lanes_index)
        results['lane_fields'].append('control_points')

        # img = results['img'].copy()
        # gt_lanes = interpolate_lanes
        # for lane in gt_lanes:
        #     for i in range(len(lane) - 1):
        #         point0 = (int(lane[i][0]), int(lane[i][1]))
        #         point1 = (int(lane[i+1][0]), int(lane[i+1][1]))
        #         cv2.line(img, point0, point1, color=(255, 0, 0), thickness=5)
        #
        # for lane_id in range(len(control_points_list)):
        #     cur_control_points = control_points_list[lane_id]  # (N_control_points, 2)
        #     for i in range(len(cur_control_points)):
        #         p = cur_control_points[i]
        #         cv2.circle(img, center=(int(p[0]), int(p[1])), radius=3, color=(0, 0, 255), thickness=-1)
        #
        # cv2.imwrite("control_points1.png", img)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(order={self.order})'
        return repr_str


@PIPELINES.register_module()
class GenerateBezierInfo:
    def __init__(self, order=3, num_sample_points=100, norm=True):
        self.order = order
        self.bezier_curve = BezierCurve(order=order)
        self.num_sample_points = num_sample_points
        self.norm = norm

    def normalize_points(self, points, img_shape):
        """
        :param points: (N_lanes, N_control_points, 2)  2: (x, y)
        :param img_shape: (h, w)
        :return:

        """
        h, w = img_shape
        points[..., 0] = points[..., 0] / w
        points[..., 1] = points[..., 1] / h
        return points

    def denormalize_points(self, points, img_shape):
        """
        :param points: (N_lanes, N_control_points, 2)  2: (x, y)
        :param img_shape: (h, w)
        :return:

        """
        h, w = img_shape
        points[..., 0] = points[..., 0] * w
        points[..., 1] = points[..., 1] * h
        return points

    def get_valid_points(self, points):
        """
        :param points: (N_lanes, N_sample_points, 2)
        :return:
            valid_mask: (N_lanes, num_sample_points)
        """
        return (points[..., 0] >= 0) * (points[..., 0] < 1) * (points[..., 1] >= 0) * (points[..., 1] < 1)

    def cubic_bezier_curve_segment(self, control_points, sample_points):
        """
            控制点在做增广时可能会超出图像边界，因此使用DeCasteljau算法裁剪贝塞尔曲线段.
            具体做法是：通过判断采样点是否超出边界来确定t的边界值，即最小边界t0和最大边界t1.
            然后便可以通过文章公式（10）计算裁剪后的控制点坐标, 并以此作为label.
        :param control_points: (N_lanes, N_controls, 2)
        :param sample_points: (N_lanes, num_sample_points, 2)
        :return:
            res: (N_lanes, N_controls, 2)
        """
        if len(control_points) == 0:
            return control_points

        N_lanes, N_sample_points = sample_points.shape[:-1]
        valid_mask = self.get_valid_points(sample_points)  # (N_lanes, num_sample_points)
        min_id = np.argmax(valid_mask + np.flip(np.arange(0, N_sample_points), axis=0), axis=-1)    # (N_lanes, )
        max_id = np.argmax(valid_mask + np.arange(0, N_sample_points), axis=-1)     # (N_lanes, )

        t = np.linspace(0.0, 1.0, num=N_sample_points, dtype=np.float32)
        t0 = t[min_id]
        t1 = t[max_id]

        # Generate transform matrix (old control points -> new control points = linear transform)
        u0 = 1 - t0  # (N_lanes, )
        u1 = 1 - t1  # (N_lanes, )
        transform_matrix_c = [np.stack([u0 ** (3 - i) * u1 ** i for i in range(4)], axis=-1),  # (N_lanes, 4)
                              np.stack([3 * t0 * u0 ** 2,
                                        2 * t0 * u0 * u1 + u0 ** 2 * t1,
                                        t0 * u1 ** 2 + 2 * u0 * u1 * t1,
                                        3 * t1 * u1 ** 2], axis=-1),
                              np.stack([3 * t0 ** 2 * u0,
                                        t0 ** 2 * u1 + 2 * t0 * t1 * u0,
                                        2 * t0 * t1 * u1 + t1 ** 2 * u0,
                                        3 * t1 ** 2 * u1], axis=-1),
                              np.stack([t0 ** (3 - i) * t1 ** i for i in range(4)], axis=-1)]
        transform_matrix_c = np.stack(transform_matrix_c, axis=-1)   # (N_lanes, 4, 4)
        res = np.matmul(transform_matrix_c, control_points)

        return res

    def cubic_bezier_curve_segmentv2(self, control_points, sample_points):
        """
            控制点在做增广时可能会超出图像边界，因此需要裁剪贝塞尔曲线段.
            具体做法是：通过判断采样点是否超出边界来找到有效的采样点，然后对这些采样点重新进行拟合.(高阶的DeCasteljau太难推)
        :param control_points: (N_lanes, N_controls, 2)
        :param sample_points: (N_lanes, num_sample_points, 2)
        :return:
            res: (N_lanes, N_controls, 2)
        """
        if len(control_points) == 0:
            return control_points,

        N_lanes, N_sample_points = sample_points.shape[:-1]
        valid_mask = self.get_valid_points(sample_points)  # (N_lanes, num_sample_points)
        min_id = np.argmax(np.flip(np.arange(0, N_sample_points), axis=0) * valid_mask, axis=-1)    # (N_lanes, )
        max_id = np.argmax(np.arange(0, N_sample_points) * valid_mask, axis=-1)     # (N_lanes, )

        control_points_list = []
        keep_indices = []
        for lane_id in range(N_lanes):
            cur_min_id = min_id[lane_id]
            cur_max_id = max_id[lane_id]
            if cur_max_id - cur_min_id < 2:
                continue
            if (cur_max_id - cur_min_id + 1) == N_sample_points:
                new_control_points = control_points[lane_id]
            else:
                valid_sample_points = sample_points[lane_id][cur_min_id:cur_max_id+1, :]      # (N_valid, 2)
                new_control_points = self.bezier_curve.get_control_points_with_fixed_endpoints(valid_sample_points)
            control_points_list.append(new_control_points)
            keep_indices.append(lane_id)

        if len(control_points_list):
            res = np.stack(control_points_list, axis=0)
        else:
            res = np.zeros((0, self.order+1, 2), dtype=np.float32)
        return res, keep_indices

    def _transform_annotation(self, results):
        img_shape = results['img_shape']    # (h, w, 3)
        h, w = img_shape[:2]
        gt_labels = results['gt_labels']
        control_points = np.array(results['control_points'], dtype=np.float32)   # (N_lanes, N_control_points, 2)
        if len(control_points) > 0:
            if self.norm:
                control_points = self.normalize_points(control_points, (h, w))

            # (N_lanes, N_sample_points, 2)   2: (x, y)
            sample_points = self.bezier_curve.get_sample_points(control_points_matrix=control_points,
                                                                num_sample_points=self.num_sample_points)

            if self.order == 3:
                control_points = self.cubic_bezier_curve_segment(control_points, sample_points)
            else:
                control_points, keep_indices = self.cubic_bezier_curve_segmentv2(control_points, sample_points)
                gt_labels = gt_labels[keep_indices]

        # img = results['img']
        # img_norm_cfg = results['img_norm_cfg']
        # mean = img_norm_cfg['mean']
        # std = img_norm_cfg['std']
        # img = mmcv.imdenormalize(img, mean, std, to_bgr=True).astype(np.uint8)
        # sample_points = self.denormalize_points(sample_points, (h, w))
        # for lane_id in range(len(sample_points)):
        #     cur_sample_points = sample_points[lane_id]  # (N_sample_points, 2)
        #     for i in range(len(cur_sample_points)-1):
        #         p1 = (int(cur_sample_points[i][0]), int(cur_sample_points[i][1]))
        #         p2 = (int(cur_sample_points[i+1][0]), int(cur_sample_points[i+1][1]))
        #         cv2.line(img, p1, p2, color=(255, 0, 0), thickness=3, lineType=cv2.LINE_AA)
        #
        # control_points = self.denormalize_points(control_points, (h, w))
        # for lane_id in range(len(control_points)):
        #     cur_control_points = control_points[lane_id]  # (N_control_points, 2)
        #     for i in range(len(cur_control_points)):
        #         p = cur_control_points[i]
        #         cv2.circle(img, center=(int(p[0]), int(p[1])), radius=3, color=(0, 0, int(255/4 * (i+1))), thickness=-1)
        #
        # cv2.imwrite("lane.png", img)

        results['gt_control_points'] = DC(to_tensor(control_points), stack=False)  # (1, hm_h=img_h/16, hm_w=img_w/16)
        results['gt_labels'] = DC(to_tensor(gt_labels), stack=False)

    def __call__(self, results):
        self._transform_annotation(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(order={self.order}, '
        repr_str += f'num_sample_points={self.num_sample_points})'
        return repr_str
