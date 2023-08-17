from __future__ import division
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import os.path as osp
import cv2
from mmcv import Config
from mmlane.datasets import build_dataset, build_dataloader
from mmlane.models import build_detector
import numpy as np
import torch
import random
from mmcv.parallel import MMDataParallel
from mmcv.image import tensor2imgs
import mmcv
from mmcv.runner.checkpoint import load_checkpoint
from mmlane.core.visualization import imshow_det_lanes


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
setup_seed(1024)

cfg_file = "../configs/laneatt/laneatt_r18_culane.py"
cfg = Config.fromfile(cfg_file)


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

cfg.data_root = '../' + cfg.data_root
cfg.data.train.data_root = cfg.data_root
cfg.data.val.data_root = cfg.data_root

datasets = build_dataset(cfg.data.train)
dataloader = build_dataloader(dataset=datasets, samples_per_gpu=8, workers_per_gpu=0, num_gpus=1, dist=False, seed=None,
                              shuffle=True)

# datasets = build_dataset(cfg.data.val)
# dataloader = build_dataloader(dataset=datasets, samples_per_gpu=1, workers_per_gpu=0, num_gpus=1, dist=False, seed=None,
#                               shuffle=False)


model = build_detector(
    cfg.model,
    train_cfg=cfg.get('train_cfg'),
    test_cfg=cfg.get('test_cfg'))

# model.init_weights()
# cfg.load_from = "/home/zichen/Documents/Project/LaneDet/mmLaneDet/work_dirs/sparse_lane/llamas/r18/default1/best_mF1_epoch_13.pth"
# if cfg.load_from:
#     load_checkpoint(model, cfg.load_from)

model = MMDataParallel(
            model.cuda(0), device_ids=[0])


# train
for i, batch_data in enumerate(dataloader):
    out = model.train_step(batch_data, None)
    loss = out['loss']
    print(out)
    break
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    # loss.backward()
    #
    # for name, param in model.named_parameters():
    #     if param.grad is None:
    #         print(name)


# model.eval()
# print(len(dataloader))
# for i, data in enumerate(dataloader):
#     print("!!!")
#     # img_filename = data['img_metas'][0].data[0][0]['filename']
#     # print(img_filename)
#     # img = cv2.imread(img_filename)
#     # cv2.imwrite(f"./results/img_{i}.png", img)
#     # with torch.no_grad():
#     #     result = model(return_loss=False, rescale=True, **data)
#     # print(result)
#     pass
#
#     # imshow_det_lanes(img, result[0], out_file=f'./results/r{i}.png', lane_color=None, score_thr=0.0)



