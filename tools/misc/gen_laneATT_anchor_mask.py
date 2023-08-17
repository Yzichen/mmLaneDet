import random
import argparse
import cv2
import torch
import numpy as np
from tqdm import trange
import os
from mmcv import Config
from mmlane.datasets import build_dataset, build_dataloader
from mmlane.models import build_detector
from mmcv.parallel import MMDataParallel
from tqdm import tqdm


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.benchmark = True #for accelerating the running
setup_seed(0)


def get_anchors_use_frequency(dataloader, model, t_pos=30., t_neg=35.):
    anchors_frequency = torch.zeros(len(model.lane_head.anchors), dtype=torch.int32)
    nb_unmatched_targets = 0
    for i, batch_data in enumerate(tqdm(dataloader)):
        gt_lanes = batch_data['gt_lanes'].data[0][0]
        gt_labels = batch_data['gt_labels'].data[0][0]

        gt_lanes = gt_lanes.cuda()
        gt_labels = gt_labels.cuda()

        n_targets = len(gt_lanes)
        if n_targets == 0:
            continue

        assigned_gt_inds, _ = model.lane_head.assign(model.lane_head.anchors,
                                                     gt_lanes,
                                                     gt_labels,
                                                     t_pos=t_pos,
                                                     t_neg=t_neg)

        positives_mask = assigned_gt_inds > 0
        negatives_mask = assigned_gt_inds == 0

        pos_inds = torch.nonzero(
            positives_mask, as_tuple=False).squeeze(-1)    # (N_pos, )
        pos_assigned_gt_inds = assigned_gt_inds[pos_inds].unique()
        n_matches = len(pos_assigned_gt_inds)

        nb_unmatched_targets += n_targets - n_matches
        assert (n_targets - n_matches) >= 0
        anchors_frequency += positives_mask.cpu()

    return anchors_frequency


def save_mask(cfg, out_file):
    datasets = build_dataset(cfg.data.train)
    dataloader = build_dataloader(dataset=datasets, samples_per_gpu=1, workers_per_gpu=0, num_gpus=1, dist=False, seed=None,
                                  shuffle=True)

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    frequency = get_anchors_use_frequency(dataloader, model, t_pos=30., t_neg=35.)
    torch.save(frequency, out_file)


def view_mask(cfg):
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    img = model.lane_head.draw_anchors()
    cv2.imwrite("anchor.png", img)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute anchor frequency for later use as anchor mask")
    parser.add_argument("--cfg_file", default="configs/laneatt/laneatt_r18_seasky.py",
                        help="Config file (e.g., `config.yml`")
    parser.add_argument("--output", default="./seasky_anchors_freq.pt", help="Output path (e.g., `anchors_mask.pt`")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    cfg_file = args.cfg_file
    cfg = Config.fromfile(args.cfg_file)

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(cfg_file)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                plg_lib = importlib.import_module(_module_path)

    save_mask(cfg, args.output)
    # view_mask(cfg)
