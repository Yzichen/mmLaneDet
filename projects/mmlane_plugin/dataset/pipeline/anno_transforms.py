import cv2
import numpy as np
from mmlane.datasets.builder import PIPELINES
import scipy.interpolate as spi
from scipy.interpolate import InterpolatedUnivariateSpline
import random
import math
from mmcv.parallel import DataContainer as DC
from mmlane.datasets.pipelines.formating import to_tensor
from shapely.geometry import Polygon, Point, LineString, MultiLineString
import copy
import mmcv


@PIPELINES.register_module()
class GenerateLaneLineV2(object):
    def __init__(self, num_points, with_theta=False):
        self.num_points = num_points
        self.n_strips = num_points - 1
        self.with_theta = with_theta
        self.idx = 0

    def filter_lane(self, lane):
        assert lane[-1][1] <= lane[0][1]
        filtered_lane = []
        used = set()
        for p in lane:
            if p[1] not in used:
                filtered_lane.append(p)
                used.add(p[1])

        return filtered_lane

    def extend_lane(self, lane, dis=10):
        """
        Args:
            lane: [(x0, y0), (x1,y1), ...]
        """
        extended = copy.deepcopy(lane)
        start = lane[1]
        end = lane[0]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        norm = math.sqrt(dx ** 2 + dy ** 2)
        dx = dx / norm
        dy = dy / norm
        extend_point = (start[0] + dx * dis, start[1] + dy * dis)
        extended.insert(0, extend_point)
        return extended

    def clamp_line(self, line, box, min_length=0):
        """
        Args:
            line: [(x0, y0), (x1,y1), ...]
            bbx: [0, 0, w-1, h-1]
            min_length: int
        """
        left, top, right, bottom = box
        loss_box = Polygon([[left, top], [right, top], [right, bottom],
                            [left, bottom]])
        line_coords = np.array(line).reshape((-1, 2))
        if line_coords.shape[0] < 2:
            return None
        try:
            line_string = LineString(line_coords)
            I = line_string.intersection(loss_box)
            if I.is_empty:
                return None
            if I.length < min_length:
                return None
            if isinstance(I, LineString):
                pts = list(I.coords)
                return pts
            elif isinstance(I, MultiLineString):
                pts = []
                Istrings = list(I)
                for Istring in Istrings:
                    pts += list(Istring.coords)
                return pts
        except:
            return None

    def sample_lane(self, points, sample_ys, img_w):
        """
        :param points: List[(x0, y0), (x1, y1), ...]  absolute   bottom --> top
        :param sample_ys: [img_h, ..., 0]
        :return:
            xs_outside_image: (N_outside, )
            xs_inside_image: (N_inside, )
        """
        # this function expects the points to be sorted
        points = np.array(points)
        if not np.all(points[1:, 1] < points[:-1, 1]):
            raise Exception('Annotaion points have to be sorted')
        x, y = points[:, 0], points[:, 1]

        # interpolate points inside domain
        assert len(points) > 1
        interp = InterpolatedUnivariateSpline(y[::-1], x[::-1], k=min(3, len(points) - 1))
        domain_min_y = y.min()
        domain_max_y = y.max()
        sample_ys_inside_domain = sample_ys[(sample_ys >= domain_min_y) & (sample_ys <= domain_max_y)]
        assert len(sample_ys_inside_domain) > 0
        interp_xs = interp(sample_ys_inside_domain)

        # extrapolate lane to the bottom of the image with a straight line using the 2 points closest to the bottom
        n_closest = max(len(points) // 5, 2)
        two_closest_points = points[:n_closest]
        extrap = np.polyfit(two_closest_points[:, 1], two_closest_points[:, 0], deg=1)
        extrap_ys = sample_ys[sample_ys > domain_max_y]
        extrap_xs = np.polyval(extrap, extrap_ys)
        all_xs = np.hstack((extrap_xs, interp_xs))

        # separate between inside and outside points
        inside_mask = (all_xs >= 0) & (all_xs < img_w)
        xs_inside_image = all_xs[inside_mask]
        xs_outside_image = all_xs[~inside_mask]

        return xs_outside_image, xs_inside_image

    def _transform_annotation(self, results):
        img_h, img_w = results['img_shape'][:2]
        max_lanes = results['max_lanes']
        strip_size = img_h / self.n_strips
        offsets_ys = np.arange(img_h, -1, -strip_size)

        old_lanes = results['gt_lanes']     # old_lanes: [[(x_00,y0), (x_01,y1), ...], [(x_10,y0), (x_11,y1), ...], ...]
        old_labels = results['gt_labels']       # old_labels: [0, 0, ...]

        # removing lanes with less than 2 points
        vaild_mask = []
        filtered_lanes = []
        for i, lane in enumerate(old_lanes):
            # 截断超出范围的line
            lane = self.clamp_line(lane.copy(), box=[0, 0, img_w - 1, img_h - 1], min_length=1)
            if lane is not None:
                vaild_mask.append(True)
                filtered_lanes.append(lane)
            else:
                vaild_mask.append(False)
        # print(old_lanes, len(old_lanes))
        # print(old_labels, len(old_labels))
        # print(vaild_mask)
        filtered_labels = old_labels[vaild_mask]

        # sort lane points by Y (bottom to top of the image)    # 图像底部-->顶部
        filtered_lanes = [sorted(lane, key=lambda x: -x[1]) for lane in filtered_lanes]
        # remove points with same Y (keep first occurrence)
        filtered_lanes = [self.filter_lane(lane) for lane in filtered_lanes]

        # for vis
        # import cv2
        # img_copy = results['img'].copy()
        # img_norm_cfg = results['img_norm_cfg']
        # mean = img_norm_cfg['mean']
        # std = img_norm_cfg['std']
        # img_vis = mmcv.imdenormalize(img_copy, mean, std, to_bgr=True).astype(np.uint8)
        # for lane in filtered_lanes:
        #     for i in range(len(lane) - 1):
        #         cv2.line(img_vis, (int(lane[i][0]), int(lane[i][1])), (int(lane[i+1][0]), int(lane[i+1][1])),
        #                  color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        # cv2.imwrite("lane.png", img_vis)

        anno_len = 1 + 1 + 1 + self.num_points
        if self.with_theta:
             anno_len += 1

        lanes = []
        lanes_labels = []
        lanes_endpoints = []
        lanes_startpoints = []
        for lane_idx, lane in enumerate(filtered_lanes):
            new_lane = np.ones(anno_len, dtype=np.float32) * -1e5
            lanes_endpoint = np.zeros(2, dtype=np.float32)
            if lane_idx >= max_lanes:
                break
            try:
                xs_outside_image, xs_inside_image = self.sample_lane(lane, offsets_ys, img_w=img_w)
            except AssertionError:
                continue
            if len(xs_inside_image) <= 1:
                continue

            all_xs = np.hstack((xs_outside_image, xs_inside_image))
            new_lane[0] = len(xs_outside_image) / self.n_strips       # normalized start_y
            new_lane[1] = xs_inside_image[0]    # img_w 尺度下的 start_x

            start = 2
            if self.with_theta:
                # thetas = []
                # for i in range(1, len(xs_inside_image)):
                #     theta = math.atan(
                #         i * strip_size /
                #         (xs_inside_image[i] - xs_inside_image[0] + 1e-5)) / math.pi
                #     theta = theta if theta > 0 else 1 - abs(theta)
                #     thetas.append(theta)
                # theta_far = sum(thetas) / len(thetas)

                thetas = []
                for i in range(1, len(lane)):
                    theta = math.atan(
                        (abs(lane[i][1] - lane[0][1]))/
                        (lane[i][0] - lane[0][0] + 1e-5)) / math.pi
                    theta = theta if theta > 0 else 1 - abs(theta)
                    thetas.append(theta)
                theta_far = sum(thetas) / len(thetas)

                new_lane[start] = theta_far     # normalized theta
                start += 1

            new_lane[start] = len(xs_inside_image)   # length
            new_lane[start+1:start+1+len(all_xs)] = all_xs     # img_w 尺度下的 xs

            lanes_endpoint[0] = (len(all_xs) - 1) / self.n_strips
            lanes_endpoint[1] = xs_inside_image[-1]

            lanes.append(new_lane)
            lanes_labels.append(filtered_labels[lane_idx])
            lanes_endpoints.append(lanes_endpoint)
            # lanes_startpoints.append(lane[0])
            lanes_startpoints.append((xs_inside_image[0], offsets_ys[len(xs_outside_image)]))   # 改动 06.05

        # for vis
        # import cv2
        # img_copy = results['img'].copy()
        # img_norm_cfg = results['img_norm_cfg']
        # mean = img_norm_cfg['mean']
        # std = img_norm_cfg['std']
        # img_copy = mmcv.imdenormalize(img_copy, mean, std, to_bgr=False).astype(np.uint8)
        # # cv2.imwrite("img.png", img_copy)
        # for i, lane in enumerate(lanes):
        #     # lane: (start_y, start_x, theta, length, coordinates(S))
        #     if lane[1]:
        #         start_y = int(lane[0] * self.n_strips)
        #         if self.with_theta:
        #             length = lane[3]
        #         else:
        #             length = lane[2]
        #         end_y = int(start_y + length - 1)
        #         theta = lane[2]
        #         assert end_y <= self.num_points, f"start_y = {start_y}, end_y = {end_y}"
        #
        #         xs = lane[-self.num_points:]
        #         valid_ys = offsets_ys[start_y:end_y]
        #         valid_xs = xs[start_y:end_y]
        #         for j in range(1, len(valid_ys)):
        #             cv2.line(img_copy, (round(valid_xs[j-1]), round(valid_ys[j-1])),
        #                      (round(valid_xs[j]), round(valid_ys[j])), (255, 0, 0), thickness=2)
        #
        #     start_point = lanes_startpoints[i]
        #     cv2.circle(img_copy, (int(start_point[0]), int(start_point[1])), radius=3, color=(0, 0, 255), thickness=-1)
        #
        #     l = 100
        #     s_x, s_y = start_point[0], start_point[1]
        #     t_x = l * math.cos(theta * math.pi)
        #     t_y = l * math.sin(theta * math.pi)
        #     e_x, e_y = int(s_x + t_x), int(s_y - t_y)
        #     cv2.line(img_copy, (int(s_x), int(s_y)), (e_x, e_y), color=(0, 0, 255), thickness=2)
        #
        # cv2.imwrite(f"./results/test{self.idx}.png", img_copy)
        # self.idx += 1

        if lanes:   # 该图像中包含车道线
            # 4+S: 1 start_y (normalized), 1 start_x (absolute), 1 theta, 1 length (absolute),
            #          S coordinates(absolute)
            results['gt_lanes'] = np.array(lanes, dtype=np.float32)         # (num_lanes, 3+S/4+S)
            results['gt_labels'] = np.array(lanes_labels, dtype=np.long)    # (num_lanes, )
            results['lane_endpoints'] = np.array(lanes_endpoints, dtype=np.float32)     # (num_lanes, 2)
            results['start_points'] = np.array(lanes_startpoints, dtype=np.float32)    # (num_lanes, 2)
        else:       # 该图像中不包含车道线
            results['gt_lanes'] = np.zeros((0, anno_len), dtype=np.float32)         # (num_lanes, 3+S/4+S)
            results['gt_labels'] = np.zeros([], dtype=np.long)    # (num_lanes, )
            results['lane_endpoints'] = np.zeros((0, 2), dtype=np.float32)     # (num_lanes, 2)
            results['start_points'] = np.zeros((0, 2), dtype=np.float32)       # (num_lanes, 2)  2: (x, y)

        for key in ['gt_lanes', 'gt_labels', 'start_points']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]), stack=False)

    def __call__(self, results):
        self._transform_annotation(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(num_points={self.num_points}, '
        repr_str += f'with_theta={self.with_theta}, '
        return repr_str