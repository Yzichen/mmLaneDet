import os
import os.path as osp
import json
import random
from .base_dataset import BaseDataset
from .builder import DATASETS
import tempfile
import numpy as np
from mmlane.core.evaluation.tusimple_metric import LaneEval
from mmcv.utils.logging import print_log
from mmlane.core.lane import Lane
import mmcv


@DATASETS.register_module
class TusimpleDataset(BaseDataset):
    def __init__(self, data_root, anno_files, pipeline, img_prefix='', seg_prefix='',
                 test_mode=False):
        super(TusimpleDataset, self).__init__(data_root, anno_files, pipeline, img_prefix, seg_prefix, test_mode)
        self.h_samples = list(range(160, 720, 10))
        self.img_w, self.img_h = 1280, 720

    def load_annotations(self):
        data_infos = []
        max_lanes = 0
        for anno_file in self.anno_files:
            anno_file = osp.join(self.data_root, anno_file)
            with open(anno_file, 'r') as anno_obj:
                lines = anno_obj.readlines()
            for line in lines:
                data = json.loads(line)
                y_samples = data['h_samples']   # [y0, y1, ...]
                gt_lanes = data['lanes']        # [[x_00, x_01, ...], [x_10, x_11, ...], ...]
                lanes = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
                # lanes: [[(x_00,y0), (x_01,y1), ...], [(x_10,y0), (x_11,y1), ...], ...]
                lanes = [lane for lane in lanes if len(lane) > 0]
                lanes_labels = [0 for lane in lanes]    # 只有一个类别
                max_lanes = max(max_lanes, len(lanes))
                lane_exist = data['lane_exist']
                data_infos.append({
                    'img_name': data['raw_file'],
                    'seg_img_name': data['seg_path'],
                    'lanes': lanes,
                    'lanes_labels': np.array(lanes_labels, dtype=np.long),
                    'lane_exist': np.array(lane_exist, dtype=np.long)
                })

        # if not self.test_mode:
        #     random.shuffle(data_infos)
        self.max_lanes = max_lanes
        return data_infos

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
            seg_map=info['seg_img_name'],
            lane_exist=info['lane_exist']
        )
        return anns_results

    def pred2lanes(self, pred):
        """
        :param pred: List[Lane0, Lane1, ...]
                 or  List[(N0, 2),  (N1, 2), ...]
        :return:
        """
        if len(pred) and isinstance(pred[0], Lane):    #  List[Lane0, Lane1, ...]
            ys = np.array(self.h_samples) / self.img_h
            lanes = []
            for lane in pred:
                xs = lane(ys)
                invalid_mask = xs < 0
                # Todo 这里应该用img_w 还是 img_w-1
                lane = (xs * self.img_w).astype(int)
                lane[invalid_mask] = -2
                lanes.append(lane.tolist())
        else:       # List[(N0, 2),  (N1, 2), ...]
            lanes = []
            for lane in pred:
                lane = lane.astype(np.int)
                lanes.append(lane[:, 0].tolist())

        return lanes

    def pred2tusimpleformat(self, idx, pred, runtime):
        runtime *= 1000.  # s to ms
        img_name = self.data_infos[idx]['img_name']
        lanes = self.pred2lanes(pred)
        output = {'raw_file': img_name, 'lanes': lanes, 'run_time': runtime}
        return json.dumps(output)

    def save_tusimple_predictions(self, predictions, filename, runtimes=None):
        if runtimes is None:
            runtimes = np.ones(len(predictions)) * 1.e-3
        lines = []
        for idx, (prediction, runtime) in enumerate(zip(predictions,
                                                        runtimes)):
            line = self.pred2tusimpleformat(idx, prediction, runtime)
            lines.append(line)
        with open(filename, 'w') as output_file:
            output_file.write('\n'.join(lines))

    def evaluate(self, results, logger=None, jsonfile_prefix=None, runtimes=None, **kwargs):
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix=jsonfile_prefix)
        gt_file = osp.join(self.data_root, self.anno_files[0])
        result_str, ret_dict = LaneEval.bench_one_submit(result_files, gt_file)
        if tmp_dir is not None:
            tmp_dir.cleanup()
        # print_log(result_str, logger=logger)
        return ret_dict

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        assert isinstance(results, list), 'results must be a list'
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
        result_files = osp.join(jsonfile_prefix,
                                     'tusimple_predictions.json')
        self.save_tusimple_predictions(results, result_files)
        return result_files, tmp_dir

    def results2json(self, predictions, outfile_prefix):
        def pred_to_lane(pred):
            out = []
            if len(pred) and isinstance(pred[0], Lane):
                ys = np.array(self.h_samples) / self.img_h
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
            else:
                for lane in pred:
                    out.append(lane.tolist())
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

                if len(pred) > 0 and isinstance(pred[0], Lane) and \
                        pred[0].metadata.get('anchor', None) is not None:
                    metadatas = extract_metadatas(pred)
                    data['meta_datas'] = metadatas

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