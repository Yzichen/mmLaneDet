import numpy as np
import torch
from scipy.special import comb as n_over_k
import math


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


class BezierCurve:
    def __init__(self, order=3, num_sample_points=36, sample_mode='UD', SID_scale=20):
        self.order = order
        self.num_control_points = order + 1
        self.sample_mode = sample_mode
        self.SID_scale = SID_scale
        self.c_matrix = self.get_bernstein_matrix(num_sample_points)

    def get_bezier_coefficient(self):
        # n: 表示控制点的数量;  t: [0, 1]的采样点, k: 表示控制点的索引(在公式中为i=0,1, ..., n)
        # 系数Bkn(t) = t**k * (1-t)**(n-k) * Cni;    Cni = n!/(k!(n-k)!)
        Mtk = lambda n, t, k: t ** k * (1 - t) ** (n - k) * n_over_k(n, k)
        BezierCoeff = lambda ts: [[Mtk(self.num_control_points - 1, t, k) for k in range(self.num_control_points)]
                                  for t in ts]

        return BezierCoeff

    def get_bernstein_matrix(self, num_sample_points, sample_mode='UD'):
        """
        Args:
            num_sample_points: int
        Returns:
            c_matrix: (N_sample_points, N_control_points)
                      分别对应num_sample_points个采样点（不同t）的num_control_points个控制点的系数.
        """
        if sample_mode == 'UD':
            t = np.linspace(0, 1, num_sample_points)    # (N_sample_points, )
        elif sample_mode == 'LID':
            bin_size = (1 - 0) / (num_sample_points * (num_sample_points - 1))
            t = np.array([0 + bin_size * i * (i + 1) for i in range(num_sample_points)])
        elif sample_mode == 'SID':
            t = np.array([(math.exp(math.log(1) + (math.log((1 + self.SID_scale) / 1) * i / num_sample_points)) - 1)
                          / self.SID_scale for i in range(num_sample_points)])
        else:
            NotImplementedError
        c_matrix = np.array(self.get_bezier_coefficient()(t), dtype=np.float32)     # (N_sample_points, N_control_points)
        return c_matrix

    def get_sample_points(self, control_points_matrix, num_sample_points=None):
        """
        Args:
            control_points_matrix: (N_lanes, N_control_points, 2)  2: (x, y)
            num_sample_points: int
        Returns:
            sample_points: (N_lanes, N_sample_points, 2)
        """
        if num_sample_points is not None:
            c_matrix = self.get_bernstein_matrix(num_sample_points)     # (N_sample_points, N_control_points)
            c_matrix, _ = check_numpy_to_torch(c_matrix)
        else:
            c_matrix, _ = check_numpy_to_torch(self.c_matrix)
        control_points_matrix, is_numpy = check_numpy_to_torch(control_points_matrix)
        c_matrix = c_matrix.to(control_points_matrix.device)

        if len(control_points_matrix) == 0:
            return np.zeros((0, num_sample_points, 2), dtype=np.float32) if is_numpy else \
                torch.zeros((0, num_sample_points, 2), dtype=torch.float32, device=control_points_matrix.device)

        # (N_sample_points, N_control_points) @ (N_lanes, N_control_points, 2)
        # --> (N_lanes, N_sample_points, 2)
        sample_points = c_matrix.matmul(control_points_matrix)
        sample_points = sample_points.numpy() if is_numpy else sample_points
        return sample_points

    def get_control_points(self, lanes, to_list=False):
        """
        :param lanes: (N, 2)
        :return:
            control_points: List[(x0, y0), (x1, y1), ...]
        """
        x, y = lanes[:, 0], lanes[:, 1]
        dy = y[1:] - y[:-1]
        dx = x[1:] - x[:-1]
        dt = (dx ** 2 + dy ** 2) ** 0.5
        t = dt / dt.sum()
        t = np.hstack(([0], t))
        t = t.cumsum()
        pseudoinverse = np.linalg.pinv(self.get_bezier_coefficient()(t))  # (N, 4) -> (4, N)
        control_points = pseudoinverse.dot(lanes)   # (4, N) @ (N, 2) --> (4, 2)
        if to_list:
            control_points = control_points.tolist()
        return control_points

    def get_control_points_with_fixed_endpoints(self, lanes, to_list=False):
        """
        :param lanes: (N, 2)
        :return:
            control_points: List[(x0, y0), (x1, y1), ...] or (N_controls, 2)
        """
        x, y = lanes[:, 0], lanes[:, 1]     # (N, )
        dy = y[1:] - y[:-1]
        dx = x[1:] - x[:-1]
        dt = (dx ** 2 + dy ** 2) ** 0.5
        t = dt / dt.sum()
        t = np.hstack(([0], t))
        t = t.cumsum()
        c_matrix = np.array(self.get_bezier_coefficient()(t), dtype=np.float32)    # (N, 4)

        c_matrix_start_end = c_matrix[:, [0, -1]]   # (N, 2)
        control_points_start_end = lanes[[0, -1], :]    # (2, 2)

        c_matrix_middle = c_matrix[:, 1:-1]   # (N, 2)
        pseudoinverse = np.linalg.pinv(c_matrix_middle)   # (2, N)

        # (2, N) @ (N, 2) --> (2, 2)
        control_points_middle = pseudoinverse.dot(lanes - c_matrix_start_end.dot(control_points_start_end))
        control_points = np.concatenate((control_points_start_end[0:1], control_points_middle,
                                         control_points_start_end[1:2]), axis=0)   # (4, 2)
        if to_list:
            control_points = control_points.tolist()
        return control_points