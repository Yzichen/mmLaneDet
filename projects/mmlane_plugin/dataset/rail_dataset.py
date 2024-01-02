import os
import os.path as osp
import json
import random

import mmcv

from mmlane.datasets.base_dataset import BaseDataset
from mmlane.datasets.builder import DATASETS
import tempfile
import numpy as np
from mmcv.utils.logging import print_log
import pickle as pkl
from ..core.evaluation import rail_metric


CATEGORYS = {
    'normal': 'list/test_split/test0_normal.txt',
    'curved': 'list/test_split/test1_curved.txt',
    'occluded': 'list/test_split/test2_occluded.txt',
    'night': 'list/test_split/test3_night.txt',
    'rainy': 'list/test_split/test4_rainy.txt',
    'snowy': 'list/test_split/test5_snowy.txt',
}


@DATASETS.register_module
class RailDataset(BaseDataset):
    def __init__(self, data_root, anno_files, pipeline, img_prefix='', seg_prefix='', test_mode=False):
        # 这里的anno_files是List[list_txt_file, ...]
        self.cls_names = ['ego_left', 'ego_right']
        super(RailDataset, self).__init__(data_root, anno_files, pipeline, img_prefix, seg_prefix, test_mode)
        self.h_samples = list(range(400, 1080, 10))
        self.img_w, self.img_h = 1920, 1080

    def load_annotations(self):
        os.makedirs('cache', exist_ok=True)
        cache_path = 'cache/rail_{}.pkl'.format('train' if not self.test_mode else 'test')
        if osp.exists(cache_path):
            with open(cache_path, 'rb') as cache_file:
                data_infos = pkl.load(cache_file)
                self.max_lanes = max(
                    len(anno['lanes']) for anno in data_infos)
                if not self.test_mode:
                    random.shuffle(data_infos)
                return data_infos

        max_lanes = 0
        data_infos = []
        for list_txt_file in self.anno_files:
            list_txt_path = osp.join(self.data_root, list_txt_file)
            with open(list_txt_path) as f:
                for line in f:
                    info = self.parse_annotation(line.split())
                    # infos: dict
                    # {
                    #     'img_name': 图像文件相对于root的相对路径
                    #     'img_path': 图像文件的绝对路径
                    #     'seg_img_path': 分割文件的绝对路径
                    #     'lane_exist': (4,)
                    #     'lanes': List[List[(x0, y0), (x1, y1), ...], List[(x0, y0), (x1, y1), ...], ...]
                    #     'lanes_labels': List[0, 0, 0, ...]
                    # }
                    if len(info['lanes']) > max_lanes:
                        max_lanes = len(info['lanes'])
                    data_infos.append(info)
        self.max_lanes = max_lanes

        # cache data infos to file
        with open(cache_path, 'wb') as cache_file:
            pkl.dump(data_infos, cache_file)

        if not self.test_mode:
            random.shuffle(data_infos)
        return data_infos

    def parse_annotation(self, line):
        """
        Args:
            line: img_path mask_path exist_lane
        Returns:
            infos: dict{
                'img_name': 图像文件相对于root的相对路径
                'img_path': 图像文件的绝对路径
                'seg_img_path': 分割文件的绝对路径
                'lane_exist': (4, )
                'lanes': List[List[(x0, y0), (x1, y1), ...], List[(x0, y0), (x1, y1), ...], ...]
                'lanes_labels': List[0, 0, 0, ...]
            }
        """
        infos = {}
        img_line = line[0]      # 图像文件相对于root的相对路径
        img_line = img_line[1 if img_line[0] == '/' else 0::]
        img_path = os.path.join(self.data_root, img_line)   # 图像文件的绝对路径
        infos['img_name'] = img_line
        if len(line) > 1:
            mask_line = line[1]     # 分割文件相对于root的相对路径
            mask_line = mask_line[1 if mask_line[0] == '/' else 0::]
            infos['seg_img_name'] = mask_line

        if len(line) > 2:
            exist_list = [int(l) for l in line[2:]]
            infos['lane_exist'] = np.array(exist_list)   # (4, )

        # remove sufix jpg and add lines.txt
        anno_path = img_path[:-3] + 'lines.txt'     # 注释文件的绝对路径
        data = []
        lane_cls = []
        with open(anno_path, 'r') as anno_file:
            for line in anno_file.readlines():
                cls_name, lane_pos = line.split()[0], line.split()[1:]
                lane_pos = list(map(float, lane_pos))
                data.append(lane_pos)
                lane_cls.append(cls_name)

        # data: List[List[x0, y0, x1, y1, ...], List[x0, y0, x1, y1, ...], ...]
        ori_lanes = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)
                  if lane[i] >= 0 and lane[i + 1] >= 0] for lane in data]
        # lanes: List[List[(x0, y0), (x1, y1), ...], List[(x0, y0), (x1, y1), ...], ...]
        ori_lanes = [list(set(lane)) for lane in ori_lanes]  # remove duplicated points

        lanes = [lane for lane in ori_lanes
                 if len(lane) > 2]  # remove lanes with less than 2 points
        lane_cls = [lane_cls[i] for i, lane in enumerate(ori_lanes) if len(lane) > 2]

        lanes = [sorted(lane, key=lambda x: x[1])
                 for lane in lanes]  # sort by y   按y轴坐标从小到大排列
        infos['lanes'] = lanes      # lanes: List[List[(x0, y0), (x1, y1), ...], List[(x0, y0), (x1, y1), ...], ...]

        lanes_labels = [self.cls_names.index(cls_name) for cls_name in lane_cls]
        infos['lanes_labels'] = np.array(lanes_labels, dtype=np.long)

        return infos

    def get_data_info(self, index):
        info = self.data_infos[index]
        img_filename = info['img_name']
        input_dict = dict(
            max_lanes=self.max_lanes,
            img_info=dict(filename=img_filename),
        )
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]
        anns_results = dict(
            lanes=info['lanes'],
            lanes_labels=info['lanes_labels'],
            seg_map=info.get('seg_img_name', None),
            lane_exist=info.get('lane_exist', None)
        )
        return anns_results

    def get_prediction_string(self, pred):
        ys = np.array(self.h_samples) / self.img_h
        out = []
        if isinstance(pred, dict):      # 如果保留了multi-layers的结果，eval的时候只利用最后一层.
            num_layers = len(pred)
            pred = pred[num_layers-1]

        for lane in pred:
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * self.img_w
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * self.img_h
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            lane_str = [
                '{:.5f} {:.5f}'.format(x, y) for x, y in zip(lane_xs, lane_ys)
            ]
            # conf = lane.metadata['conf']
            # lane_str.insert(0, '{:.3f}'.format(conf))

            lane_str = ' '.join(lane_str)
            if lane_str != '':
                out.append(lane_str)

        return '\n'.join(out)

    def save_culane_predictions(self, predictions, culane_pred_dir):
        for idx, pred in enumerate(predictions):
            output_dir = osp.join(culane_pred_dir, osp.dirname(self.data_infos[idx]['img_name']))
            output_filename = osp.basename(self.data_infos[idx]['img_name'])[:-3] + 'lines.txt'
            os.makedirs(output_dir, exist_ok=True)
            lanes_pred_str = self.get_prediction_string(pred)
            with open(osp.join(output_dir, output_filename), 'w') as f:
                f.write(lanes_pred_str)

    def evaluate(self, results, logger=None, jsonfile_prefix=None, runtimes=None, **kwargs):
        culane_pred_dir, tmp_dir = self.format_results(results, jsonfile_prefix=jsonfile_prefix)

        ret_dict = {}
        # for cate, cate_file in CATEGORYS.items():
        #     result = rail_metric.eval_predictions(culane_pred_dir,
        #                                             self.data_root,
        #                                             os.path.join(self.data_root, cate_file),
        #                                             iou_thresholds=[0.5],
        #                                             official=True,
        #                                             logger=logger,
        #                                             img_shape=(1080, 1920, 3))
        #     ret_dict[f'{cate}_F1@50'] = result[0.5]['F1']

        result = rail_metric.eval_predictions(culane_pred_dir,
                                                self.data_root,
                                                osp.join(self.data_root, self.anno_files[0]),
                                                iou_thresholds=np.linspace(0.5, 0.95, 10),
                                                official=True,
                                                logger=logger,
                                                img_shape=(1080, 1920, 3))

        ret_dict['F1@50'] = result[0.5]['F1']
        ret_dict['F1@75'] = result[0.75]['F1']
        ret_dict['mF1'] = result['mean']['F1']
        if tmp_dir is not None:
            tmp_dir.cleanup()
        # print_log(result_str, logger=logger)
        return ret_dict

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        assert isinstance(results, list), 'results must be a list '
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            self.results2json(results, jsonfile_prefix)  # 也保存json格式.
            tmp_dir = None

        if not osp.exists(jsonfile_prefix):
            os.mkdir(jsonfile_prefix)

        culane_pred_dir = jsonfile_prefix
        self.save_culane_predictions(results, culane_pred_dir)
        return culane_pred_dir, tmp_dir

    def results2json(self, predictions, outfile_prefix):
        def pred_to_lane(pred):
            ys = np.array(self.h_samples) / self.img_h
            out = []
            for lane in pred:
                xs = lane(ys)
                valid_mask = (xs >= 0) & (xs < 1)
                xs = xs * self.img_w
                lane_xs = xs[valid_mask]
                lane_ys = ys[valid_mask] * self.img_h
                lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
                lane = [(x, y) for x, y in zip(lane_xs, lane_ys)]
                if len(lane) >= 2:
                    out.append(lane)
            return out

        def extract_metadatas(pred):
            metadatas = []
            for lane in pred:
                metadatas.append(lane.metadata)
            return metadatas

        lane_json_results = []
        for idx, pred in enumerate(predictions):
            if isinstance(pred, list):  # List[lane0, lane1, ...]
                pred_lane = pred_to_lane(pred)  # List[[(x0, y0), (x1, y1), ...], [(x0, y0), (x1, y1), ...], ...]
                data = dict()
                data['pred_lane'] = pred_lane
                data['img_name'] = self.data_infos[idx]['img_name']

                if len(pred) > 0:
                    metadatas = extract_metadatas(pred)
                    data['meta_datas'] = metadatas
                else:
                    data['meta_datas'] = None

                lane_json_results.append(data)
            elif isinstance(pred, dict):    # {layer_id: List[lane0, lane1, ...]; ...}
                preds_dict = dict()
                for layer_id, cur_pred in pred.items():
                    pred_lane = pred_to_lane(cur_pred)  # List[[(x0, y0), (x1, y1), ...], [(x0, y0), (x1, y1), ...], ...]
                    preds_dict[layer_id] = pred_lane
                data = dict()
                data['pred_lane'] = preds_dict
                data['img_name'] = self.data_infos[idx]['img_name']
                lane_json_results.append(data)
            else:
                raise NotImplementedError

        out_json_file = osp.join(outfile_prefix, 'pred_lane.json')
        mmcv.dump(lane_json_results, out_json_file)
