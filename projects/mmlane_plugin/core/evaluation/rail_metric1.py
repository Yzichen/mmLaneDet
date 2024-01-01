import os
import argparse
from functools import partial
import numpy as np
from matplotlib import pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm
from p_tqdm import t_map, p_map
from scipy.interpolate import splprep, splev
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, Polygon
import logging


class Lane:
    def __init__(self, points):
        self.points = points
        self.matched = False


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


def discrete_cross_iou(xs, ys, width=30, img_shape=(1080, 1920, 3)):
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


def continuous_cross_iou(xs, ys, width=30, img_shape=(1080, 1920, 3)):
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
                  img_shape=(1080, 1920, 3)):
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

    # print("pred: ", pred)
    # print("anno: ", anno)

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
    _error = []
    for thr in iou_thresholds:
        tp = int((ious[row_ind, col_ind] > thr).sum())   # 正确预测的数量
        fp = len(pred) - tp     # 错误预测的数量
        fn = len(anno) - tp     # 漏减的数量
        _metric[thr] = [tp, fp, fn]

        for k in range(len(row_ind)):
            pred_id = row_ind[k]
            gt_id = col_ind[k]
            if ious[pred_id, gt_id] > thr:
                pred_lane = pred[pred_id]   # (N_p_pred, 2)
                anno_lane = anno[gt_id]     # (N_p_gt, 2)
                for gt_x, gt_y in anno_lane:
                    find = False
                    for pred_x, pred_y in pred_lane:
                        if pred_y == gt_y:
                            x_error = abs(gt_x - pred_x)
                            _error.append(x_error)
                            find = True
                    if not find:
                        _error.append(5.0)

        _metric[thr].append(_error)

    return _metric


def load_culane_img_data(path, anno=False):
    """
    :param path: path of lines.txt
    :return:
        img_data: List[[(x0, y0), (x1, y1), ...], [(x0, y0), (x1, y1), ...], ...]
    """
    with open(path, 'r') as data_file:
        img_data = data_file.readlines()    # List[str(cls, x0 y0 x1 y1 ...), str(cls, x0 y0 x1 y1 ...), ...]

    if not anno:
        conf = [line.split()[0] for line in img_data]
        conf = list(map(float, conf))
    else:
        conf = [1.0 for line in img_data]

    img_data = [line.split()[1:] for line in img_data]
    img_data = [list(map(float, lane)) for lane in img_data]    # List[[x0, y0, x1, y1, ...], [x0, y0, x1, y1, ...], ...]
    img_data = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)]
                for lane in img_data]   # List[[(x0, y0), (x1, y1), ...], [(x0, y0), (x1, y1), ...], ...]
    img_data1 = []
    conf1 = []
    for i, lane in enumerate(img_data):
        if len(lane) >= 2:
            img_data1.append(lane)
            conf1.append(conf[i])

    return img_data1, conf1


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
    confs = []
    for i, path in enumerate(filepaths):
        img_data, conf = load_culane_img_data(path, anno)   # List[[(x0, y0), (x1, y1), ...], [(x0, y0), (x1, y1), ...], ...]
        data.append(img_data)
        confs.append(conf)

    return data, confs

def draw_pr(rec, prec):
    plt.plot(rec, prec, label='PR curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('Precision-Recall')
    plt.legend()
    plt.savefig(f'pr.pdf', dpi=1000)


def eval_predictions(pred_dir,
                     anno_dir,
                     list_path,
                     iou_thresholds=np.linspace(0.5, 0.95, 10),
                     width=30,
                     official=True,
                     sequential=False,
                     logger=None,
                     img_shape=(1080, 1920, 3)):
    if logger is None:
        logger = logging.getLogger(__name__)
    logger.info('Calculating metric for List: {}'.format(list_path))

    data, confs = load_culane_data(pred_dir, list_path)
    predictions = []
    pred_confs = []
    img_ids = []
    for img_id in range(len(data)):
        pred_lanes = data[img_id]   # List[[(x0, y0), (x1, y1), ...], [(x0, y0), (x1, y1), ...], ...]
        conf = confs[img_id]
        for lane_id in range(len(pred_lanes)):
            predictions.append(pred_lanes[lane_id])     # List[(x0, y0), (x1, y1), ...]
            pred_confs.append(conf[lane_id])     # float
            img_ids.append(img_id)

    assert len(img_ids) == len(predictions) == len(pred_confs)

    data, _ = load_culane_data(anno_dir, list_path, anno=True)
    annos = dict()
    npos = 0
    for img_id in range(len(data)):
        gt_lanes = data[img_id]     # List[[(x0, y0), (x1, y1), ...], [(x0, y0), (x1, y1), ...], ...]
        gt_lanes1 = []
        for gt_lane in gt_lanes:
            gt_lanes1.append(Lane(points=gt_lane))
            npos += 1
        annos[img_id] = gt_lanes1

    pred_confs = np.array(pred_confs)
    sorted_ind = np.argsort(-pred_confs)
    sorted_scores = np.sort(-pred_confs)
    sorted_preds = []
    for id in sorted_ind:
        sorted_preds.append(predictions[id])
    img_ids = [img_ids[i] for i in sorted_ind]

    # go down dets and mark TPs and FPs
    # APs = dict()
    # for iou_thres in iou_thresholds:

    nd = len(img_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        print("d = ", d)
        cur_gt_lanes = annos[img_ids[d]]  # List[Lane0, Lane1, ...]
        anno = [gt_lanes.points for gt_lanes in cur_gt_lanes]

        pred = [sorted_preds[d]]

        interp_pred = np.array([interp(pred_lane, n=5) for pred_lane in pred],
                           dtype=object)  # (N_pred=1, N_points, 2)
        interp_anno = np.array([interp(anno_lane, n=5) for anno_lane in anno],
                               dtype=object)  # (N_gt, N_points, 2)

        # ious: (N_pred=1, N_gt),  pred车道线与gt车道线之间的IoU.
        ious = discrete_cross_iou(interp_pred,  # (N_pred, N_points, 2)
                                  interp_anno,  # (N_gt, N_points, 2)
                                  width=width,
                                  img_shape=img_shape)
        iou = ious[0]  # (N_gt)
        iou_max = np.max(iou)  # 最大重叠
        max_id = np.argmax(iou)  # 最大重合率对应的gt

        if iou_max > 0.5:
            cuLane = cur_gt_lanes[max_id]
            if not cuLane.matched:
                tp[d] = 1
                cuLane.matched = True
            else:
                fp[d] = 1
        else:
            fp[d] = 1

    # compute precision recall
    fp = np.cumsum(fp)  # np.cumsum() 按位累加
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    pr_dict = {'prec': prec, 'rec': rec}
    np.save('bezier.npy', pr_dict)

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    draw_pr(mrec, mpre)

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    AP = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    # APs[iou_thres] = AP
    print(AP)
    return AP


def main():
    args = parse_args()
    for list_path in args.list:
        results = eval_predictions(args.pred_dir,
                                   args.anno_dir,
                                   list_path,
                                   width=args.width,
                                   official=args.official,
                                   sequential=args.sequential)
        print(results)

def parse_args():
    parser = argparse.ArgumentParser(description="Measure CULane's metric")
    parser.add_argument(
        "--pred_dir",
        default='/home/zichen/Documents/Project/LaneDet/Temp/mmLaneDet/work_dirs/bezier/seasky/epoch_400/results_conf',
        help="Path to directory containing the predicted lanes",
        )
    parser.add_argument(
        "--anno_dir",
        default='/home/zichen/Documents/Dataset/202Data/ImageSet/Seasky_video',
        help="Path to directory containing the annotated lanes",
        )
    parser.add_argument("--width",
                        type=int,
                        default=30,
                        help="Width of the lane")
    parser.add_argument("--list",
                        default=['/home/zichen/Documents/Dataset/202Data/ImageSet/Seasky_video/list/test.txt'],
                        nargs='+',
                        help="Path to txt file containing the list of files",
                        )
    parser.add_argument("--sequential",
                        default=False,
                        help="Run sequentially instead of in parallel")
    parser.add_argument("--official",
                        default=True,
                        help="Use official way to calculate the metric")

    return parser.parse_args()


if __name__ == '__main__':
    main()
