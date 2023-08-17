import os
import os.path as osp
import pickle as pkl
import json
import random
from tqdm import tqdm
from .base_dataset import BaseDataset
from .builder import DATASETS
import tempfile
import numpy as np
from .llamas_utils import get_horizontal_values_for_four_lanes
import mmcv
import cv2
import mmlane.core.evaluation.llamas_metric as llamas_metric


TEST_IMGS_DIR = 'color_images/test'
SPLIT_DIRECTORIES = {'train': 'labels/train', 'val': 'labels/valid'}

@DATASETS.register_module
class LLAMASDataset(BaseDataset):
    def __init__(self, data_root, split, pipeline, anno_files=None, img_prefix='', seg_prefix='',
                 test_mode=False):
        self.split = split
        if split != 'test' and split not in SPLIT_DIRECTORIES.keys():
            raise Exception('Split `{}` does not exist.'.format(split))
        if split != 'test':
            self.labels_dir = os.path.join(data_root, SPLIT_DIRECTORIES[split])
        self.img_w, self.img_h = 1276, 717
        super(LLAMASDataset, self).__init__(data_root, anno_files, pipeline, img_prefix, seg_prefix, test_mode)

        self.h_samples = list(range(300, 717, 1))

    def get_json_paths(self):
        json_paths = []
        for root, _, files in os.walk(self.labels_dir):
            for file in files:
                if file.endswith(".json"):
                    json_paths.append(os.path.join(root, file))
        return json_paths

    def get_img_path(self, json_path):
        base_name = '/'.join(json_path.split('/')[1:])
        image_path = os.path.join(
            'color_images', base_name.replace('.json', '_color_rect.png'))
        return image_path

    def load_annotations(self):
        # the labels are not public for the test set yet
        if self.split == 'test':
            data_infos = []
            imgs_dir = os.path.join(self.data_root, TEST_IMGS_DIR)
            for root, _, files in os.walk(imgs_dir):
                for file in files:
                    if file.endswith('.png'):
                        img_path = os.path.join(root, file)
                        rel_img_path = os.path.relpath(img_path, start=self.data_root)
                        data_infos.append({
                            'img_name': rel_img_path
                        })
            self.max_lanes = 4
            return data_infos

        # Waiting for the dataset to load is tedious, let's cache it
        os.makedirs('cache', exist_ok=True)
        cache_path = 'cache/llamas_{}.pkl'.format(self.split)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as cache_file:
                data_infos = pkl.load(cache_file)
                self.max_lanes = max(
                    len(anno['lanes']) for anno in data_infos)
                if not self.test_mode:
                    random.shuffle(data_infos)
                return data_infos

        max_lanes = 0
        data_infos = []
        print("Searching annotation files...")
        json_paths = self.get_json_paths()
        print('{} annotations found.'.format(len(json_paths)))
        for json_path in tqdm(json_paths):
            print(json_path)
            lanes = get_horizontal_values_for_four_lanes(json_path)
            lanes = [[(x, y) for x, y in zip(lane, range(self.img_h))
                      if x >= 0] for lane in lanes]
            lanes = [lane for lane in lanes if len(lane) > 0]
            lanes = [list(set(lane))
                     for lane in lanes]  # remove duplicated points
            lanes = [lane for lane in lanes
                     if len(lane) > 2]  # remove lanes with less than 2 points

            lanes = [sorted(lane, key=lambda x: x[1])
                     for lane in lanes]  # sort by y
            lanes.sort(key=lambda lane: lane[0][0])
            lanes_labels = [0 for lane in lanes]  # 只有一个类别

            mask_path = json_path.replace('.json', '.png')
            # generate seg labels
            seg = np.zeros((self.img_h, self.img_w, 3))
            for i, lane in enumerate(lanes):
                for j in range(0, len(lane) - 1):
                    cv2.line(seg, (round(lane[j][0]), lane[j][1]),
                             (round(lane[j + 1][0]), lane[j + 1][1]),
                             (i + 1, i + 1, i + 1),
                             thickness=15)
            cv2.imwrite(mask_path, seg)

            relative_json_path = osp.relpath(json_path, start=self.data_root)
            relative_img_path = self.get_img_path(relative_json_path)
            relative_seg_path = osp.relpath(mask_path, start=self.data_root)

            max_lanes = max(max_lanes, len(lanes))
            data_infos.append({
                'img_name': relative_img_path,
                'seg_img_name': relative_seg_path,
                'lanes': lanes,
                'lanes_labels': np.array(lanes_labels, dtype=np.long),
            })

        self.max_lanes = max_lanes

        # cache data infos to file
        with open(cache_path, 'wb') as cache_file:
            pkl.dump(data_infos, cache_file)
        if not self.test_mode:
            random.shuffle(data_infos)
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
            seg_map=info.get('seg_img_name', None),
            lane_exist=info.get('lane_exist', None)
        )
        return anns_results

    def get_prediction_string(self, pred):
        ys = np.array(self.h_samples) / (self.img_h - 1)
        out = []
        for lane in pred:
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * (self.img_w - 1)
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * (self.img_h - 1)
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            lane_str = ' '.join([
                '{:.5f} {:.5f}'.format(x, y) for x, y in zip(lane_xs, lane_ys)
            ])
            if lane_str != '':
                out.append(lane_str)

        return '\n'.join(out)

    def save_llamas_predictions(self, predictions, llamas_pred_dir):
        for idx, pred in enumerate(predictions):
            img_path = self.data_infos[idx]['img_name']
            output_filename = '/'.join(img_path.split('/')[-2:]).replace(
                '_color_rect.png', '.lines.txt')
            output_filepath = osp.join(llamas_pred_dir, output_filename)
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            output = self.get_prediction_string(pred)
            with open(output_filepath, 'w') as out_file:
                out_file.write(output)

    def evaluate(self, results, logger=None, jsonfile_prefix=None, runtimes=None, **kwargs):
        llamas_pred_dir, tmp_dir = self.format_results(results, jsonfile_prefix=jsonfile_prefix)

        result = llamas_metric.eval_predictions(llamas_pred_dir,
                                                self.labels_dir,
                                                iou_thresholds=np.linspace(0.5, 0.95, 10),
                                                unofficial=False,
                                                logger=logger)
        ret_dict = {}
        ret_dict['F1@50'] = result[0.5]['F1']
        ret_dict['F1@75'] = result[0.75]['F1']
        ret_dict['mF1'] = result['mean']['F1']
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

        llamas_pred_dir = jsonfile_prefix
        self.save_llamas_predictions(results, llamas_pred_dir)
        return llamas_pred_dir, tmp_dir

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

                if pred[0].metadata.get('anchor', None) is not None:
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