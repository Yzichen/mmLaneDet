import os
import argparse
from functools import partial

import cv2
import numpy as np
from tqdm import tqdm
from p_tqdm import t_map, p_map
from scipy.interpolate import splprep, splev
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, Polygon
import logging


def draw_lane(lane, img=None, img_shape=None, width=30):
    """
    :param lane: (N_points, 2)
    :param img: None
    :param img_shape: (H, W, 3)
    :param width: int
    :return:
        img: (H, W, 3)
    """
    if img is None:
        img = np.zeros(img_shape, dtype=np.uint8)   # (H, W, 3)
    lane = lane.astype(np.int32)    # (N_points, 2)
    for p1, p2 in zip(lane[:-1], lane[1:]):
        cv2.line(img,
                 tuple(p1),     # (x1, y1)
                 tuple(p2),     # (x2, y2)
                 color=(255, 255, 255),
                 thickness=width)
    return img


def discrete_cross_iou(xs, ys, width=30, img_shape=(590, 1640, 3)):
    """
    :param xs: (N_pred, N_points, 2)  预测车道线
    :param ys: (N_gt, N_points, 2)    gt车道线
    :param width: int
    :param img_shape: (H, W, 3)
    :return:
        ious: (N_pred, N_gt)
    """
    # List[(H, W, 3), (H, W, 3), ...]   len = N_pred
    xs = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in xs]
    # List[(H, W, 3), (H, W, 3), ...]   len = N_gt
    ys = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in ys]

    ious = np.zeros((len(xs), len(ys)))     # (N_pred, N_gt)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ious[i, j] = (x & y).sum() / (x | y).sum()
    return ious


def continuous_cross_iou(xs, ys, width=30, img_shape=(590, 1640, 3)):
    h, w, _ = img_shape
    image = Polygon([(0, 0), (0, h - 1), (w - 1, h - 1), (w - 1, 0)])
    xs = [
        LineString(lane).buffer(distance=width / 2., cap_style=1,
                                join_style=2).intersection(image)
        for lane in xs
    ]
    ys = [
        LineString(lane).buffer(distance=width / 2., cap_style=1,
                                join_style=2).intersection(image)
        for lane in ys
    ]

    ious = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ious[i, j] = x.intersection(y).area / x.union(y).area

    return ious


def interp(points, n=50):
    """
    :param points: List[(x0, y0), (x1, y1), ...]
    :param n: 应该表示B-样条的控制点
    :return:
    """
    x = [x for x, _ in points]      # List[x0, x1, x2, ...]
    y = [y for _, y in points]      # List[y0, y1, y2, ...]
    # (t,c,k):  a tuple containing the vector of knots, the B-spline coefficients, and the degree of the spline.
    # u: An array of the values of the parameter.
    tck, u = splprep([x, y], s=0, t=n, k=min(3, len(points) - 1))

    u = np.linspace(0., 1., num=(len(u) - 1) * n + 1)
    return np.array(splev(u, tck)).T


def culane_metric(pred,
                  anno,
                  width=30,
                  iou_thresholds=[0.5],
                  official=True,
                  img_shape=(590, 1640, 3)):
    """
    :param pred: List[[(x0, y0), (x1, y1), ...], [(x0, y0), (x1, y1), ...], ....]
    :param anno: List[[(x0, y0), (x1, y1), ...], [(x0, y0), (x1, y1), ...], ....]
    :param width: int
    :param iou_thresholds: List[iou0, iou1, ...]
    :param official:  Bool
    :param img_shape: (H, W, 3)
    :return:
        _metric: dict{
            iou_thr0: [tp, fp, fn],
            iou_thr1: [tp, fp, fn],
            ...
        }
    """
    _metric = {}
    for thr in iou_thresholds:
        tp = 0
        fp = 0 if len(anno) != 0 else len(pred)     # 如果该场景没有目标(len(anno)==0)，所有pred都是FP; 如果有目标，则FP初始化为0
        fn = 0 if len(pred) != 0 else len(anno)     # 如果该场景没有预测(len(pred)==0)，则FN(漏检)为该场景的真实目标数量.
        _metric[thr] = [tp, fp, fn]

    interp_pred = np.array([interp(pred_lane, n=5) for pred_lane in pred],
                           dtype=object)  # (N_pred, N_points, 2)
    interp_anno = np.array([interp(anno_lane, n=5) for anno_lane in anno],
                           dtype=object)  # (N_gt, N_points, 2)

    if official:
        # ious: (N_pred, N_gt),  pred车道线与gt车道线之间的IoU.
        ious = discrete_cross_iou(interp_pred,  # (N_pred, N_points, 2)
                                  interp_anno,  # (N_gt, N_points, 2)
                                  width=width,
                                  img_shape=img_shape)
    else:
        ious = continuous_cross_iou(interp_pred,
                                    interp_anno,
                                    width=width,
                                    img_shape=img_shape)

    # row_ind:(N_pos, ),  col_ind:(N_pos, )
    row_ind, col_ind = linear_sum_assignment(1 - ious)

    _metric = {}
    for thr in iou_thresholds:
        tp = int((ious[row_ind, col_ind] > thr).sum())   # 正确预测的数量
        fp = len(pred) - tp     # 错误预测的数量
        fn = len(anno) - tp     # 漏减的数量
        _metric[thr] = [tp, fp, fn]
    return _metric


def load_culane_img_data(path, anno=False):
    """
    :param path: path of lines.txt
    :return:
        img_data: List[[(x0, y0), (x1, y1), ...], [(x0, y0), (x1, y1), ...], ...]
    """
    with open(path, 'r') as data_file:
        img_data = data_file.readlines()    # List[str(cls, x0 y0 x1 y1 ...), str(cls, x0 y0 x1 y1 ...), ...]
    if anno:
        img_data = [line.split()[1:] for line in img_data]      # 第一个数据是cls name.
    else:
        img_data = [line.split() for line in img_data]  # 第一个数据是cls name.

    img_data = [list(map(float, lane)) for lane in img_data]    # List[[x0, y0, x1, y1, ...], [x0, y0, x1, y1, ...], ...]
    img_data = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)]
                for lane in img_data]   # List[[(x0, y0), (x1, y1), ...], [(x0, y0), (x1, y1), ...], ...]
    img_data = [lane for lane in img_data if len(lane) >= 2]

    return img_data


def load_culane_data(data_dir, file_list_path, anno=False):
    """
    :param data_dir:
    :param file_list_path:
    :return:
        data: List[List[[(x0, y0), (x1, y1), ...], [(x0, y0), (x1, y1), ...], ...],
                   List[[(x0, y0), (x1, y1), ...], [(x0, y0), (x1, y1), ...], ...],
                  ...
              ]
    """
    with open(file_list_path, 'r') as file_list:
        filepaths = [
            os.path.join(
                data_dir, line[1 if line[0] == '/' else 0:].rstrip().replace(
                    '.jpg', '.lines.txt')) for line in file_list.readlines()
        ]   # List[txt_path0, txt_path1, ...]

    data = []
    for path in filepaths:
        img_data = load_culane_img_data(path, anno)   # List[[(x0, y0), (x1, y1), ...], [(x0, y0), (x1, y1), ...], ...]
        data.append(img_data)

    return data


def eval_predictions(pred_dir,
                     anno_dir,
                     list_path,
                     iou_thresholds=[0.5],
                     width=30,
                     official=True,
                     sequential=False,
                     logger=None,
                     img_shape=(590, 1640, 3)):
    if logger is None:
        logger = logging.getLogger(__name__)
    logger.info('Calculating metric for List: {}'.format(list_path))
    predictions = load_culane_data(pred_dir, list_path)
    annotations = load_culane_data(anno_dir, list_path, anno=True)
    if sequential:
        results = map(
            partial(culane_metric,
                    width=width,
                    official=official,
                    iou_thresholds=iou_thresholds,
                    img_shape=img_shape), predictions, annotations)
    else:
        from multiprocessing import Pool, cpu_count
        from itertools import repeat
        with Pool(cpu_count()) as p:
            results = p.starmap(culane_metric, zip(predictions, annotations,
                        repeat(width),
                        repeat(iou_thresholds),
                        repeat(official),
                        repeat(img_shape)))

    # results: List[
    #     dict{
    #         iou_thr0: [tp, fp, fn],
    #         iou_thr1: [tp, fp, fn],
    #         ...
    #     },
    #     dict{
    #         iou_thr0: [tp, fp, fn],
    #         iou_thr1: [tp, fp, fn],
    #         ...
    #     },
    #     ...
    # ]     # 对应不同样本的_metric

    mean_f1, mean_prec, mean_recall, total_tp, total_fp, total_fn = 0, 0, 0, 0, 0, 0
    ret = {}
    for thr in iou_thresholds:
        tp = sum(m[thr][0] for m in results)    # 该thr下的TP和.
        fp = sum(m[thr][1] for m in results)    # 该thr下的FP和.
        fn = sum(m[thr][2] for m in results)    # 该thr下的FN和.
        precision = float(tp) / (tp + fp) if tp != 0 else 0     # 该thr下的准确率.
        recall = float(tp) / (tp + fn) if tp != 0 else 0        # 该thr下的召回率.
        f1 = 2 * precision * recall / (precision + recall) if tp !=0 else 0     # 该thr下的F1.
        logger.info('iou thr: {:.2f}, tp: {}, fp: {}, fn: {},'
                'precision: {}, recall: {}, f1: {}'.format(
            thr, tp, fp, fn, precision, recall, f1))

        mean_f1 += f1 / len(iou_thresholds)     # 统计所有thr下的F1.
        mean_prec += precision / len(iou_thresholds)    # 统计所有thr下的准确率.
        mean_recall += recall / len(iou_thresholds)     # 统计所有thr下的召回率.
        total_tp += tp
        total_fp += fp
        total_fn += fn
        ret[thr] = {
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }
    if len(iou_thresholds) > 2:
        logger.info('mean result, total_tp: {}, total_fp: {}, total_fn: {},'
                'precision: {}, recall: {}, f1: {}'.format(total_tp, total_fp,
            total_fn, mean_prec, mean_recall, mean_f1))
        ret['mean'] = {
            'TP': total_tp,
            'FP': total_fp,
            'FN': total_fn,
            'Precision': mean_prec,
            'Recall': mean_recall,
            'F1': mean_f1
        }

    return ret


def main():
    args = parse_args()
    for list_path in args.list:
        results = eval_predictions(args.pred_dir,
                                   args.anno_dir,
                                   list_path,
                                   width=args.width,
                                   official=args.official,
                                   sequential=args.sequential)

        header = '=' * 20 + ' Results ({})'.format(
            os.path.basename(list_path)) + '=' * 20
        print(header)
        for metric, value in results.items():
            if isinstance(value, float):
                print('{}: {:.4f}'.format(metric, value))
            else:
                print('{}: {}'.format(metric, value))
        print('=' * len(header))


def parse_args():
    parser = argparse.ArgumentParser(description="Measure CULane's metric")
    parser.add_argument(
        "--pred_dir",
        help="Path to directory containing the predicted lanes",
        required=True)
    parser.add_argument(
        "--anno_dir",
        help="Path to directory containing the annotated lanes",
        required=True)
    parser.add_argument("--width",
                        type=int,
                        default=30,
                        help="Width of the lane")
    parser.add_argument("--list",
                        nargs='+',
                        help="Path to txt file containing the list of files",
                        required=True)
    parser.add_argument("--sequential",
                        action='store_true',
                        help="Run sequentially instead of in parallel")
    parser.add_argument("--official",
                        action='store_true',
                        help="Use official way to calculate the metric")

    return parser.parse_args()


if __name__ == '__main__':
    main()
