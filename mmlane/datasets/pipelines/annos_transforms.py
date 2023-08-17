import cv2
import numpy as np
from ..builder import PIPELINES
import scipy.interpolate as spi
from scipy.interpolate import InterpolatedUnivariateSpline
import random
import math
from mmcv.parallel import DataContainer as DC
from .formating import to_tensor
from shapely.geometry import Polygon, Point, LineString, MultiLineString
import copy
import PIL
import mmcv


@PIPELINES.register_module()
class GenerateLaneLine(object):
    def __init__(self, num_points, with_theta=False):
        self.num_points = num_points
        self.n_strips = num_points - 1
        self.with_theta = with_theta

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
                thetas = []
                for i in range(1, len(xs_inside_image)):
                    theta = math.atan(
                        i * strip_size /
                        (xs_inside_image[i] - xs_inside_image[0] + 1e-5)) / math.pi
                    theta = theta if theta > 0 else 1 - abs(theta)
                    thetas.append(theta)
                theta_far = sum(thetas) / len(thetas)

                # thetas = []
                # theta1 = math.atan(
                #     1 * strip_size /
                #     (xs_inside_image[1] - xs_inside_image[0] + 1e-5)) / math.pi
                # theta1 = theta1 if theta1 > 0 else 1 - abs(theta1)
                # theta2 = math.atan(
                #     (len(xs_inside_image) - 1) * strip_size /
                #     (xs_inside_image[-1] - xs_inside_image[0] + 1e-5)) / math.pi
                # theta2 = theta2 if theta2 > 0 else 1 - abs(theta2)
                # thetas.append(theta1)
                # thetas.append(theta2)
                # theta_far = sum(thetas) / len(thetas)

                new_lane[start] = theta_far     # normalized theta
                start += 1

            new_lane[start] = len(xs_inside_image)   # length
            new_lane[start+1:start+1+len(all_xs)] = all_xs     # img_w 尺度下的 xs

            lanes_endpoint[0] = (len(all_xs) - 1) / self.n_strips
            lanes_endpoint[1] = xs_inside_image[-1]

            lanes.append(new_lane)
            lanes_labels.append(filtered_labels[lane_idx])
            lanes_endpoints.append(lanes_endpoint)
            lanes_startpoints.append(lane[0])
            # lanes_startpoints.append((xs_inside_image[0], offsets_ys[len(xs_outside_image)]))   # 改动 06.05

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


@PIPELINES.register_module()
class GenerateRowAnchorLabel:
    def __init__(self, row_anchor, grid_num=100, max_lanes=6, img_scale=(800, 288)):
        self.row_anchor = row_anchor
        self.grid_num = grid_num
        self.max_lanes = max_lanes
        self.img_scale = img_scale

    def find_start_pos(self, row_anchor, start_line):
        l, r = 0, len(row_anchor)-1
        while True:
            mid = int((l+r)/2)
            if r - l == 1:
                return r
            if row_anchor[mid] < start_line:
                l = mid
            if row_anchor[mid] > start_line:
                r = mid
            if row_anchor[mid] == start_line:
                return mid

    def grid_pts(self, pts, w):
        """
        Args:
            pts: (N_lanes, N_row_anchors, 2)  2: (row_anchor, x_pos/-1)
            w:  img_w
        Returns:
            row_anchor_label: (N_row_anchors, N_lanes)
                              # 不存在lane时: --> num_cols=griding_num
                              # 存在: col_id
        """
        grid_interval = (w - 1) / (self.grid_num - 1)
        row_anchor_label = np.zeros((len(self.row_anchor), self.max_lanes))     # (N_row_anchors, N_lanes)
        for i in range(self.max_lanes):
            pti = pts[i, :, 1]      # 该lane在所有row_anchors上对应的x_pos  (N_row_anchors, ）
            row_anchor_label[:, i] = np.array([round(pt / grid_interval) if pt != -1 else
                                               self.grid_num for pt in pti])

        return row_anchor_label

    def _transform_annotation(self, results):
        seg = results['gt_semantic_seg']        # (H, W)

        h, w = seg.shape
        dst_w, dst_h = self.img_scale
        if h != dst_h:
            scale_f = lambda x: int((x * 1.0/dst_h) * h)
            row_anchor = list(map(scale_f, self.row_anchor))
        else:
            row_anchor = self.row_anchor

        # (N_lanes, N_row_anchors, 2)  2: (row_anchor, x_pos/-1)
        all_idx = np.zeros((self.max_lanes, len(row_anchor), 2))
        for i, r in enumerate(row_anchor):  # 遍历row_anchor
            cur_row_seg = seg[round(r)]     # (W, )  label图像中第r行, 与row anchor对应
            for lane_idx in range(1, self.max_lanes+1):   # 遍历lanes
                pos = np.where(cur_row_seg == lane_idx)[0]      # 寻找与该lane对应的区域（x区域）, 可能有多个点.
                if len(pos) == 0:
                    all_idx[lane_idx - 1, i, 0] = r
                    all_idx[lane_idx - 1, i, 1] = -1
                    continue
                pos = np.mean(pos)
                all_idx[lane_idx - 1, i, 0] = r     # 分别记录对应的row_anchor 和 所处的x坐标.
                all_idx[lane_idx - 1, i, 1] = pos

        # 将lanes延伸到图像底部
        all_idx_cp = all_idx.copy()     # (N_lanes, N_row_anchor, 2)
        for i in range(self.max_lanes):
            if np.all(all_idx_cp[i, :, 1] == -1):
                continue

            # 获取该lane对应的所有valid points
            valid = all_idx_cp[i, :, 1] != -1    # (N_row_anchor, )
            valid_idx = all_idx_cp[i, valid, :]  # (N_valid, 2)
            if valid_idx[-1, 0] == row_anchor[-1]:
                # 意味着这条lane已经到达图像底部
                continue
            if len(valid_idx) < 6:
                # 这条lane太短，不适合去延伸.
                continue

            valid_idx_half = valid_idx[len(valid_idx) // 2:, :]     # (N_valid//2, 2)  2: (row_anchor_id, x_pos)
            p = np.polyfit(valid_idx_half[:, 0], valid_idx_half[:, 1], deg=1)  # 根据row_anchor 和 对应的 x_pos 进行拟合.
            start_line = valid_idx_half[-1, 0]  # lane截止时的row_anchor
            pos = self.find_start_pos(row_anchor, start_line) + 1  # 找到该row_anchor在所有row_anchors中id + 1.

            fitted = np.polyval(p, row_anchor[pos:])  # 根据后N_row-pos个row_anchor值，拟合对应的的x_pos.
            fitted = np.array([-1 if x < 0 or x > w - 1 else x for x in fitted])  # 如果超出图像范围为-1, 否则为拟合的x_pos.

            assert np.all(all_idx_cp[i, pos:, 1] == -1)
            all_idx_cp[i, pos:, 1] = fitted  # 根据拟合值进行更新

        row_anchor_label = self.grid_pts(all_idx_cp, w)     # (N_row_anchors, N_lanes)
        results['row_anchor_label'] = DC(to_tensor(row_anchor_label), stack=True, pad_dims=None)

        # img = results['img'].copy()
        # for lane_idx in range(row_anchor_label.shape[1]):
        #     lane = []
        #     cur_row_anchor_label = row_anchor_label[:, lane_idx]    # (N_row_anchors, )
        #     for i, r in enumerate(row_anchor):
        #         if cur_row_anchor_label[i] == self.grid_num:
        #             continue
        #         y = r
        #         x = int(cur_row_anchor_label[i] * (w - 1) / (self.grid_num - 1))
        #         lane.append((x, y))
        #
        #     if len(lane) >= 2:
        #         for i in range(len(lane) - 1):
        #             cv2.line(img, lane[i], lane[i+1], color=(255, 0, 0), thickness=5)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)

    def __call__(self, results):
        self._transform_annotation(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(row_anchor={self.row_anchor}, '
        repr_str += f'grid_num={self.grid_num}, '
        repr_str += f'max_lanes={self.max_lanes}) '
        return repr_str


# For CondLaneDet
@PIPELINES.register_module
class GenerateCondInfo(object):
    def __init__(self,
                 down_scale,
                 hm_down_scale,
                 line_width=3,
                 max_mask_sample=5,
                 radius=6,
                 hm_radius=None,
                 offset_thres=10,
                 ):
        self.down_scale = down_scale
        self.hm_down_scale = hm_down_scale
        self.line_width = line_width
        self.max_mask_sample = max_mask_sample
        self.radius = radius
        if hm_radius is None:
            hm_radius = radius
        self.hm_radius = hm_radius
        self.offset_thres = offset_thres

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

    def draw_umich_gaussian(self, heatmap, center, radius, k=1):
        """
        Args:
            heatmap: (hm_h, hm_w)   1/16
            center: (x0', y0'),  1/16
            radius: float
        Returns:
            heatmap: (hm_h, hm_w)
        """
        def gaussian2D(shape, sigma=1):
            """
            Args:
                shape: (diameter=2*r+1, diameter=2*r+1)
            Returns:
                h: (diameter, diameter)
            """
            m, n = [(ss - 1.) / 2. for ss in shape]
            # y: (1, diameter)    x: (diameter, 1)
            y, x = np.ogrid[-m:m + 1, -n:n + 1]
            # (diameter, diameter)
            h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            return h

        diameter = 2 * radius + 1
        # (diameter, diameter)
        gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
        x, y = int(center[0]), int(center[1])
        height, width = heatmap.shape[0:2]
        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)
        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    def select_mask_points(self, ct, r, shape, max_sample=5):
        """
        从lane起始点周围选取max_sample个点, 后续应当是取这些点对应的 kernel params 进行lane shape预测, 提供正确的训练样本.
        Args:
            ct: (center_x, center_y)
            r: float
            shape: (hm_h, hm_w)
        Returns:
            sample_points: [(x0, y0), (x1, y1), ..]
        """

        def in_range(pt, w, h):
            if pt[0] >= 0 and pt[0] < w and pt[1] >= 0 and pt[1] < h:
                return True
            else:
                return False

        def cal_dis(p1, p2):
            return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        h, w = shape[:2]
        sample_points = []
        r = max(int(r // 2), 1)
        start_x, end_x = ct[0] - r, ct[0] + r
        start_y, end_y = ct[1] - r, ct[1] + r
        for x in range(start_x, end_x + 1):
            for y in range(start_y, end_y + 1):
                if x == ct[0] and y == ct[1]:
                    continue
                if in_range((x, y), w, h) and cal_dis((x, y), ct) <= r + 0.1:
                    sample_points.append([x, y])
        if len(sample_points) > max_sample - 1:
            sample_points = random.sample(sample_points, max_sample - 1)
        sample_points.append([ct[0], ct[1]])
        return sample_points

    def downscale_lane(self, lane, downscale):
        """
        :param lane: List[(x0, y0), (x1, y1), ...]
        :param downscale: int
        :return:
            downscale_lane: List[(x0/downscale, y0/downscale), (x1/downscale, y1/downscale), ...]
        """
        downscale_lane = []
        for point in lane:
            downscale_lane.append((point[0] / downscale, point[1] / downscale))
        return downscale_lane

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

    def draw_label(self,
                   mask,
                   polygon,
                   val,
                   shape_type='polygon',
                   width=3):
        polygon = copy.deepcopy(polygon)
        xy = []
        for i in range(len(polygon)):
            xy.append((polygon[i][0], polygon[i][1]))

        mask = PIL.Image.fromarray(mask)
        if shape_type == 'polygon':
            PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=val, fill=val)
        else:
            PIL.ImageDraw.Draw(mask).line(xy=xy, fill=val, width=width)
        mask = np.array(mask, dtype=np.uint8)
        return mask

    def get_line_intersection(self, y, line):
        """
        Args:
            y:
            line: [(x0, y0), (x1,y1), ...]
        """

        def in_line_range(val, start, end):
            s = min(start, end)
            e = max(start, end)
            if val >= s and val <= e and s != e:
                return True
            else:
                return False

        # def choose_min_reg(val, ref):
        #     min_val = 1e5
        #     index = -1
        #     if len(val) == 0:
        #         return None
        #     else:
        #         for i, v in enumerate(val):
        #             if abs(v - ref) < min_val:
        #                 min_val = abs(v - ref)
        #                 index = i
        #     return val[index]

        # reg_x = []
        # for i in range(len(line) - 1):
        #     point_start, point_end = line[i], line[i + 1]
        #     if in_line_range(y, point_start[1], point_end[1]):
        #         k = (point_end[0] - point_start[0]) / (
        #                 point_end[1] - point_start[1])
        #         reg_x.append(k * (y - point_start[1]) + point_start[0])
        # reg_x = choose_min_reg(reg_x, x)

        reg_x = None
        for i in range(len(line) - 1):
            point_start, point_end = line[i], line[i + 1]
            if in_line_range(y, point_start[1], point_end[1]):
                k = (point_end[0] - point_start[0]) / (
                        point_end[1] - point_start[1])
                reg_x = (k * (y - point_start[1]) + point_start[0])
                break

        return reg_x

    def _transform_annotation(self, results):
        img_shape = results['img_shape']
        img_h, img_w = img_shape[:2]
        mask_h = int(img_h // self.down_scale)      # img_h / 4  = 80
        mask_w = int(img_w // self.down_scale)      # img_w / 4  = 200
        hm_h = int(img_h // self.hm_down_scale)  # img_h / 16 = 20
        hm_w = int(img_w // self.hm_down_scale)  # img_w / 16 = 50

        # gt init
        gt_hm = np.zeros((1, hm_h, hm_w), np.float32)   # (1, hm_h=img_h/16, hm_w=img_w/16)
        gt_masks = []
        gt_lanes = results['gt_lanes']  # List[List[(x0, y0), (x1, y1), ...], List[(x0, y0), (x1, y1), ...], ...]

        # for vis
        # img = results['img']
        # img_norm_cfg = results['img_norm_cfg']
        # mean = img_norm_cfg['mean']
        # std = img_norm_cfg['std']
        # img = mmcv.imdenormalize(img, mean, std, to_bgr=False).astype(np.uint8)
        # for lane in gt_lanes:
        #     for i in range(len(lane) - 1):
        #         point0 = (int(lane[i][0]), int(lane[i][1]))
        #         point1 = (int(lane[i+1][0]), int(lane[i+1][1]))
        #         cv2.line(img, point0, point1, color=(255, 0, 0), thickness=5)
        # cv2.imwrite("lane.png", img)

        for lane_id, lane in enumerate(gt_lanes):
            # lane: List[(x0, y0), (x1, y1), ...]
            # 将lane沿图像从下到上排列 （y由大到小）
            lane = sorted(lane, key=lambda x: x[1], reverse=True)
            # 截断超出范围的line
            lane = self.clamp_line(lane, box=[0, 0, img_w-1, img_h-1], min_length=1)

            if lane is None or len(lane) <= 1:
                continue

            # 找到车道起始点坐标， 在1/16尺度下.
            gt_hm_lane_start = (lane[0][0] * 1/self.hm_down_scale, lane[0][1] * 1/self.hm_down_scale)
            gt_hm_lane_start_int = (int(gt_hm_lane_start[0]), int(gt_hm_lane_start[1]))
            # 根据车道起始点坐标，绘制gaussian heatmap.
            self.draw_umich_gaussian(gt_hm[0], gt_hm_lane_start_int, self.hm_radius)

            # 从lane起始点gt_hm_lane_start_int周围选取max_sample个点, 后续取这些点对应的kernel params 进行lane shape预测,
            # 因此这些采样点提供了正确的训练样本.
            sample_points = self.select_mask_points(
                ct=gt_hm_lane_start_int,
                r=self.radius,
                shape=(hm_h, hm_w),
                max_sample=self.max_mask_sample)

            # prediction init， 在1/4尺度下.
            gt_pos = np.zeros((mask_h, ), np.float32)  # (mask_h, )
            pos_mask = np.zeros((mask_h, ), np.float32)  # (mask_h, )
            gt_offset = np.zeros((mask_h, mask_w), np.float32)  # (1, mask_h, mask_w)
            offset_mask = np.zeros((mask_h, mask_w), np.float32)     # (1, mask_h, mask_w)
            lane_range = np.zeros((mask_h, ), np.int64)  # (1, mask_h)

            # downscaled lane: List[(x0, y0), (x1, y1), ...], 1/4 尺度.
            lane = self.downscale_lane(lane, downscale=self.down_scale)

            # 向图像底部延伸一个点.
            extended_lane = self.extend_lane(lane)
            lane_array = np.array(lane)
            y_min, y_max = np.min(lane_array[:, 1]), np.max(lane_array[:, 1])

            m = np.zeros((mask_h, mask_w), np.uint8)  # (mask_h, mask_w)
            # 根据extended_line 绘制mask map, 只有lane所在位置为1.  (mask_h, mask_w)
            polygon = np.array(extended_lane)
            polygon_map = self.draw_label(
                m, polygon, 1, 'line', width=self.line_width + 9) > 0

            for y in range(polygon_map.shape[0]):
                pos_xs = self.get_line_intersection(y, lane)  # float
                if pos_xs is None:
                    continue
                for x in np.where(polygon_map[y, :])[0]:
                    offset = pos_xs - x
                    gt_offset[y, x] = offset
                    if abs(offset) < self.offset_thres:
                        offset_mask[y, x] = 1

                    if y_min <= y <= y_max:
                        gt_pos[y] = pos_xs
                        pos_mask[y] = 1
                    lane_range[y] = 1

            gt_masks.append({
                'gt_offset': gt_offset,  # (mask_h, mask_w),  记录offset的gt.
                'offset_mask': offset_mask,  # (mask_h, mask_w) 需要监督offset 回归的mask, 只有lane附近的点对应的mask=1
                'sample_points': sample_points,  # List[[x0, y0], [x1, y1], ...], 取这些点对应的kernel params 进行lane shape预测, 提供正确的训练样本.
                'row': gt_pos,  # (mask_h), 每一行lane对应的准确的pos_x (float).
                'row_mask': pos_mask,  # (mask_h, ), 表示location预测的监督mask.
                'range': lane_range,  # (mask_h),  表明每一行是否存在车道线.
                'label': 0
            })

        #     ys = np.arange(0, mask_h, step=1)
        #     valid_y = ys[pos_mask.astype(np.bool)]
        #     valid_x = gt_pos[pos_mask.astype(np.bool)]
        #
        #
        #     for i in range(len(valid_y) - 1):
        #         point0 = (int(valid_x[i] * 8), int(valid_y[i] * 8))
        #         point1 = (int(valid_x[i+1] * 8), int(valid_y[i+1] * 8))
        #         cv2.line(img, point0, point1, color=(255, 0, 0), thickness=5)
        #
        # cv2.imwrite("lane.png", img)

        results['gt_hm'] = DC(to_tensor(gt_hm), stack=True)  # (1, hm_h=img_h/16, hm_w=img_w/16)
        results['gt_masks'] = DC(gt_masks, cpu_only=True)
        # results['down_scale'] = self.down_scale  # 4
        # results['hm_down_scale'] = self.hm_down_scale  # 16
        # results['hm_shape'] = (hm_h, hm_w)
        # results['mask_shape'] = (mask_h, mask_w)

    def __call__(self, results):
        self._transform_annotation(results)
        return results


@PIPELINES.register_module()
class GenerateGAInfo(object):
    def __init__(self,
                 radius=2,
                 fpn_cfg=dict(
                     hm_idx=0,
                     fpn_down_scale=[8, 16, 32],
                     sample_per_lane=[41, 21, 11],
                 )
                 ):
        self.radius = radius

        self.hm_idx = fpn_cfg.get('hm_idx')
        self.fpn_down_scale = fpn_cfg.get('fpn_down_scale')
        self.sample_per_lane = fpn_cfg.get('sample_per_lane')
        self.hm_down_scale = self.fpn_down_scale[self.hm_idx]
        self.fpn_layer_num = len(self.fpn_down_scale)

    def ploy_fitting_cube(self, line, h, w, sample_num=100):
        """
        Args:
            line: List[(x0, y0), (x1, y1), ...]    # y从大到小排列, 即lane由图像底部-->顶部
            h: f_H
            w: f_W
            sample_num: sample_per_lane
        Returns:
            key_points: (N_sample, 2)
        """
        line_coords = np.array(line).reshape((-1, 2))  # (N, 2)
        # y从小到大排列, 即lane由图像顶部-->底部
        line_coords = np.array(sorted(line_coords, key=lambda x: x[1]))

        lane_x = line_coords[:, 0]
        lane_y = line_coords[:, 1]

        if len(lane_y) < 2:
            return None
        new_y = np.linspace(max(lane_y[0], 0), min(lane_y[-1], h), sample_num)

        sety = set()
        nX, nY = [], []
        for i, (x, y) in enumerate(zip(lane_x, lane_y)):
            if y in sety:
                continue
            sety.add(x)
            nX.append(x)
            nY.append(y)
        if len(nY) < 2:
            return None

        if len(nY) > 3:
            ipo3 = spi.splrep(nY, nX, k=3)
            ix3 = spi.splev(new_y, ipo3)
        else:
            ipo3 = spi.splrep(nY, nX, k=1)
            ix3 = spi.splev(new_y, ipo3)
        return np.stack((ix3, new_y), axis=-1)

    def downscale_lane(self, lane, downscale):
        """
        :param lane: List[(x0, y0), (x1, y1), ...]
        :param downscale: int
        :return:
            downscale_lane: List[(x0/downscale, y0/downscale), (x1/downscale, y1/downscale), ...]
        """
        downscale_lane = []
        for point in lane:
            downscale_lane.append((point[0] / downscale, point[1] / downscale))
        return downscale_lane

    def clip_line(self, pts, h, w):
        pts_x = np.clip(pts[:, 0], 0, w - 1)[:, None]
        pts_y = np.clip(pts[:, 1], 0, h - 1)[:, None]
        return np.concatenate([pts_x, pts_y], axis=-1)

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

    def draw_umich_gaussian(self, heatmap, center, radius, k=1):
        """
        Args:
            heatmap: (hm_h, hm_w)   1/16
            center: (x0', y0'),  1/16
            radius: float
        Returns:
            heatmap: (hm_h, hm_w)
        """
        def gaussian2D(shape, sigma=1):
            """
            Args:
                shape: (diameter=2*r+1, diameter=2*r+1)
            Returns:
                h: (diameter, diameter)
            """
            m, n = [(ss - 1.) / 2. for ss in shape]
            # y: (1, diameter)    x: (diameter, 1)
            y, x = np.ogrid[-m:m + 1, -n:n + 1]
            # (diameter, diameter)
            h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            return h

        diameter = 2 * radius + 1
        # (diameter, diameter)
        gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
        x, y = int(center[0]), int(center[1])
        height, width = heatmap.shape[0:2]
        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)
        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    def _transform_annotation(self, results):
        img_h, img_w = results['img_shape'][:2]
        max_lanes = results['max_lanes']

        gt_lanes = results['gt_lanes']  # List[List[(x0, y0), (x1, y1), ...], List[(x0, y0), (x1, y1), ...], ...]

        # 遍历 fpn levels, 寻找每个车道线在该level特征图上对应采样点的位置.
        gt_hm_lanes = {}
        for l in range(self.fpn_layer_num):
            lane_points = []
            fpn_down_scale = self.fpn_down_scale[l]
            f_h = img_h // fpn_down_scale
            f_w = img_w // fpn_down_scale
            for i, lane in enumerate(gt_lanes):
                # downscaled lane: List[(x0, y0), (x1, y1), ...]
                lane = self.downscale_lane(lane, downscale=self.fpn_down_scale[l])
                # 将lane沿图像从下到上排列 （y由大到小）
                lane = sorted(lane, key=lambda x: x[1], reverse=True)
                pts = self.ploy_fitting_cube(lane, f_h, f_w, self.sample_per_lane[l])  # (N_sample, 2)
                if pts is not None:
                    pts_f = self.clip_line(pts, f_h, f_w)  # (N_sample, 2)
                    pts = np.int32(pts_f)
                    lane_points.append(pts[:, ::-1])  # (N_sample, 2)   2： (y, x)

            # (max_lane_num,  N_sample, 2)  2： (y, x)
            # 保存每个车道线在该level特征图上对应采样点的位置.
            lane_points_align = -1 * np.ones((max_lanes, self.sample_per_lane[l], 2))
            if len(lane_points) != 0:
                lane_points_align[:len(lane_points)] = np.stack(lane_points, axis=0)    # (num_lanes, N_sample, 2)
            gt_hm_lanes[l] = lane_points_align

        # 在最终所利用的level下，生成heatmap、offset等.
        # gt init
        hm_h = img_h // self.hm_down_scale
        hm_w = img_w // self.hm_down_scale
        kpts_hm = np.zeros((1, hm_h, hm_w), np.float32)     # (1, hm_H, hm_W)
        kp_offset = np.zeros((2, hm_h, hm_w), np.float32)   # (2, hm_H, hm_W)
        sp_offset = np.zeros((2, hm_h, hm_w), np.float32)   # (2, hm_H, hm_W)  key points -> start points
        kp_offset_mask = np.zeros((2, hm_h, hm_w), np.float32)  # (2, hm_H, hm_W)
        sp_offset_mask = np.zeros((2, hm_h, hm_w), np.float32)  # (2, hm_H, hm_W)

        start_points = []
        for i, lane in enumerate(gt_lanes):
            # downscaled lane: List[(x0, y0), (x1, y1), ...]
            lane = self.downscale_lane(lane, downscale=self.hm_down_scale)
            if len(lane) < 2:
                continue
            # (N_sample=int(360 / self.hm_down_scale), 2)
            lane = self.ploy_fitting_cube(lane, hm_h, hm_w, int(360 / self.hm_down_scale))
            if lane is None:
                continue
            # 将lane沿图像从下到上排列 （y由大到小）
            lane = sorted(lane, key=lambda x: x[1], reverse=True)
            lane = self.clamp_line(lane, box=[0, 0, hm_w - 1, hm_h - 1], min_length=1)
            if lane is None:
                continue

            start_point, end_point = lane[0], lane[-1]    # (2, ),  (2, )
            start_points.append(start_point)
            for pt in lane:
                pt_int = (int(pt[0]), int(pt[1]))   # (x, y)
                # 根据关键点坐标, 生成heatmap.
                self.draw_umich_gaussian(kpts_hm[0], pt_int, radius=self.radius)

                # 生成 compensation offsets, 对quantization error进行补偿.
                offset_x = pt[0] - pt_int[0]
                offset_y = pt[1] - pt_int[1]
                kp_offset[0, pt_int[1], pt_int[0]] = offset_x
                kp_offset[1, pt_int[1], pt_int[0]] = offset_y
                # 生成kp_offset_mask, 只有关键点位置处为1.
                kp_offset_mask[:, pt_int[1], pt_int[0]] = 1

                # 关键点到起始点之间的偏移
                offset_x = start_point[0] - pt_int[0]
                offset_y = start_point[1] - pt_int[1]
                sp_offset[0, pt_int[1], pt_int[0]] = offset_x
                sp_offset[1, pt_int[1], pt_int[0]] = offset_y
                # 生成kp_offset_mask, 只有关键点位置处为1.
                sp_offset_mask[:, pt_int[1], pt_int[0]] = 1

        targets = {}
        targets['gt_hm_lanes'] = gt_hm_lanes
        targets['gt_kpts_hm'] = kpts_hm
        targets['gt_kp_offset'] = kp_offset
        targets['gt_sp_offset'] = sp_offset
        targets['kp_offset_mask'] = kp_offset_mask
        targets['sp_offset_mask'] = sp_offset_mask

        # for vis
        # import cv2
        # img_copy = results['img'].copy()
        # img_norm_cfg = results['img_norm_cfg']
        # mean = img_norm_cfg['mean']
        # std = img_norm_cfg['std']
        # img_copy = mmcv.imdenormalize(img_copy, mean, std, to_bgr=False).astype(np.uint8)
        # img_vis = mmcv.imresize(img_copy, (hm_w, hm_h))
        # for sp in start_points:
        #     cv2.circle(img_vis, (int(sp[0]), int(sp[1])), color=(0, 0, 255), radius=1, thickness=-1)
        # cv2.imwrite("img.png", img_vis)
        #
        # hm = kpts_hm[0]     # (hm_h, hm_w)
        # hm_vis = (hm * 255).astype(np.uint8)
        # hm_vis = cv2.applyColorMap(hm_vis, cv2.COLORMAP_JET)
        # hm_vis = 0.5 * img_vis + 0.5 * hm_vis
        # hm_vis = hm_vis.astype(np.uint8)
        # cv2.imwrite("heatmap.png", hm_vis)
        #
        # kp_offset_mask_vis = kp_offset_mask[0]
        # valid_id = np.nonzero(kp_offset_mask_vis)
        # kp_vis = img_vis.copy()
        # kp_vis[valid_id] = (0, 0, 255)
        # kp_vis = kp_vis.astype(np.uint8)
        # cv2.imwrite("kp.png", kp_vis)
        #
        # sp_offset_mask_vis = sp_offset_mask[0]
        # sp_vis = img_vis.copy()
        # valid_id = np.nonzero(sp_offset_mask_vis)
        # kp_coord = np.stack((valid_id[1], valid_id[0]), axis=-1)        # (Nk, 2)  2:(x, y)
        # sp_offset_valid = sp_offset[:, valid_id[0], valid_id[1]]
        # sp_offset_valid = sp_offset_valid.transpose(1, 0)   # (Nk, 2)  2:(tx, ty)
        # sp_coord = kp_coord + sp_offset_valid
        # sp_coord = sp_coord.astype(np.int)
        # for i in range(len(sp_coord)):
        #     kp = kp_coord[i]
        #     sp = sp_coord[i]
        #     cv2.arrowedLine(sp_vis, (kp[0], kp[1]), (sp[0], sp[1]), color=(255, 0, 0), thickness=1)
        #
        # cv2.imwrite("sp.png", sp_vis)

        return targets

    def __call__(self, results):
        targets = self._transform_annotation(results)
        results['gt_targets'] = DC(targets, cpu_only=True)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
