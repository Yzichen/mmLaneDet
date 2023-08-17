import cv2
import numpy as np
import os
import os.path as osp
from mmlane.core.lane import Lane

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
    (255, 0, 128),
    (0, 128, 255),
    (0, 255, 128),
    (128, 255, 255),
    (255, 128, 255),
    (255, 255, 128),
    (60, 180, 0),
    (180, 60, 0),
    (0, 60, 180),
    (0, 180, 60),
    (60, 0, 180),
    (180, 0, 60),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
]


def imshow_det_lanes(img,
                     lanes,
                     sample_y=None,
                     score_thr=0.3,
                     lane_color=(255, 0, 0),
                     thickness=5,
                     win_name='',
                     show=False,
                     wait_time=0,
                     out_file=None,
                     meta_datas=None,
                     ):
    h, w = img.shape[:2]
    if sample_y is None:
        sample_y = range(h-1, 0, -10)

    if len(lanes) > 0 and isinstance(lanes[0], Lane):
        lane_arrays = []
        for lane in lanes:
            if 'conf' in lane.metadata:
                score = lane.metadata['conf']
                if score < score_thr:
                    continue
            lane_array = lane.to_array(sample_y=sample_y, img_size=(h, w))
            if len(lane_array):
                lane_arrays.append(lane_array)
    else:
        lane_arrays = [np.array(lane, dtype=np.float32) for lane in lanes]

    lanes_xys = []
    for _, lane_array in enumerate(lane_arrays):
        xys = []    # List[(x0, y0), (x1, y1), ...]
        for x, y in lane_array:
            if x <= 0 or y <= 0:
                continue
            x, y = int(x), int(y)
            xys.append((x, y))
        lanes_xys.append(xys)

    # # List[List[(x0, y0), (x1, y1), ...], List[(x0, y0), (x1, y1), ...], ...]
    sorted_id = sorted(range(len(lanes_xys)), key=lambda k: lanes_xys[k][0][0])
    lanes_xys.sort(key=lambda xys: xys[0][0])

    for idx, xys in enumerate(lanes_xys):
        for i in range(1, len(xys)):
            cv2.line(img, xys[i - 1], xys[i], lane_color if lane_color is not None else COLORS[idx], thickness=thickness,
                     lineType=cv2.LINE_AA)

        if isinstance(lanes[0], Lane) and 'conf' in lanes[idx].metadata:
            conf = lanes[idx].metadata['conf']
            start_point = xys[0]
            cv2.putText(img, f'{conf:.2f}', (start_point[0]+20, start_point[1]-5), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=1, color=lane_color if lane_color is not None else COLORS[idx], thickness=thickness)

        if meta_datas is not None:
            conf = meta_datas[idx]['conf']
            start_point = xys[0]
            cv2.putText(img, f'{conf:.2f}', (start_point[0]+20, start_point[1]-5), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=1, color=lane_color if lane_color is not None else COLORS[idx], thickness=2)

    if show:
        cv2.imshow(win_name, img)
        cv2.waitKey(wait_time)

    if out_file:
        if not osp.exists(osp.dirname(out_file)):
            os.makedirs(osp.dirname(out_file))
        cv2.imwrite(out_file, img)

    return img


def imshow_gt_det_lanes(img,
                        det_lanes,  # List[[(x0, y0), (x1, y1), ...], [(x0, y0), (x1, y1), ...], ...]
                        gt_lanes,   # List[[(x0, y0), (x1, y1), ...], [(x0, y0), (x1, y1), ...], ...]
                        sample_y=None,
                        score_thr=0.0,
                        pred_lane_color=None,
                        gt_lane_color=None,
                        thickness=5,
                        win_name='',
                        show=False,
                        wait_time=0,
                        out_file=None,
                        overlay_gt_pred=False,
                        meta_datas=None,
                        ):
    img_with_gt = imshow_det_lanes(
        img=img.copy(),
        lanes=gt_lanes,
        lane_color=gt_lane_color,
        sample_y=sample_y,
        thickness=thickness,
    )

    if overlay_gt_pred:
        img = imshow_det_lanes(
            img=img_with_gt,
            lanes=det_lanes,
            lane_color=pred_lane_color,
            sample_y=sample_y,
            thickness=thickness,
            score_thr=score_thr,
            meta_datas=meta_datas,
        )
    else:
        img_with_det = imshow_det_lanes(
            img=img.copy(),
            lanes=det_lanes,
            lane_color=pred_lane_color,
            thickness=thickness,
            score_thr=score_thr,
            meta_datas=meta_datas,
        )

        img_H, img_W = img.shape[:2]
        img = np.zeros((img_H, 2*img_W, 3), dtype=np.uint8)
        img[:, :img_W, :] = img_with_det
        img[:, img_W:, :] = img_with_gt

    if show:
        cv2.imshow(win_name, img)
        cv2.waitKey(wait_time)

    if out_file:
        if not osp.exists(osp.dirname(out_file)):
            os.makedirs(osp.dirname(out_file))
        cv2.imwrite(out_file, img)

    return img
