import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from mmcv.runner import force_fp32
from mmdet.core import reduce_mean
from mmlane.models import build_loss
from ..builder import HEADS
from .base_dense_head import BaseDenseHead
from mmlane.core import Lane


@HEADS.register_module
class SCNNHead(BaseDenseHead):
    def __init__(self,
                 num_points=64,
                 max_num_lanes=6,
                 img_size=(800, 320),
                 stride=8,
                 in_channels=128,
                 fc_hidden_dim=128,
                 loss_seg=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     class_weight=1.0,
                     bg_cls_weight=0.4,
                     loss_weight=1.0,
                     reduction='mean',
                 ),
                 loss_exist=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='mean',
                     loss_weight=0.1
                 ),
                 with_exist=True,
                 train_cfg=None,
                 test_cfg=dict(
                     exist_score_thr=0.5,
                     seg_score_thr=0.6,
                 ),
                 sync_cls_avg_factor=False,
                 init_cfg=None,
                 **kwargs):
        super(SCNNHead, self).__init__(init_cfg)
        self.n_offsets = num_points
        self.max_num_lanes = max_num_lanes
        self.img_w, self.img_h = img_size
        self.register_buffer(name='prior_ys', tensor=torch.linspace(0, 1, steps=self.n_offsets, dtype=torch.float32))

        self.use_sigmoid_seg = loss_seg.get('use_sigmoid', False)
        if self.use_sigmoid_seg:
            self.seg_num_classes = max_num_lanes
        else:
            self.seg_num_classes = max_num_lanes + 1

        self.dropout = nn.Dropout2d(0.1)
        self.seg_conv = nn.Conv2d(in_channels, self.seg_num_classes, kernel_size=1)

        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_seg.get('class_weight', None)
        if class_weight is not None and (self.__class__ is SCNNHead):
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            bg_cls_weight = loss_seg.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(max_num_lanes + 1) * class_weight
            # set background class as the last indice
            class_weight[0] = bg_cls_weight
            loss_seg.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_seg:
                loss_seg.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        self.loss_seg = build_loss(loss_seg)

        self.with_exist = with_exist
        if self.with_exist:
            self.use_sigmoid_exist = loss_exist.get('use_sigmoid', False)
            if self.use_sigmoid_exist:
                self.exist_num_classes = max_num_lanes
            else:
                self.exist_num_classes = max_num_lanes + 1

            self.exist_head = nn.Sequential(
                nn.Linear(int(self.seg_num_classes*self.img_h/(stride*2)*self.img_w/(stride*2)), fc_hidden_dim),
                nn.ReLU(),
                nn.Linear(fc_hidden_dim, self.exist_num_classes)
            )
            self.loss_exist = build_loss(loss_exist)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def forward(self, x, **kwargs):
        '''
        Args:
            x: input features (B, C=128, H3, W3)  图像的1/8尺度
            kwargs:
                dict {
                    'img': (B, 3, img_H, img_W),
                    'seg': (B, img_H, img_W),
                    'meta': List[dict0, dict1, ...]
                }
        Return:
            output:
                pred_seg: (B, seg_num_classes, H, W)
                pred_exist: (B, max_num_lanes)
        '''
        batch_size = x.shape[0]
        x = self.dropout(x)
        x = self.seg_conv(x)    # (B, seg_num_classes, H3, W3)
        seg_map = F.interpolate(x, size=(self.img_h, self.img_w),
                                mode='bilinear', align_corners=False)   # (B, seg_num_classes, H, W)
        output = {'pred_seg': seg_map}

        if self.with_exist:
            x = F.softmax(x, dim=1)      # (B, seg_num_classes, H3, W3)
            x = F.avg_pool2d(x, 2, stride=2, padding=0)     # (B, seg_num_classes, H3/2, W3/2)
            x = x.view(batch_size, -1)      # (B, seg_num_classes*(H3/2)*(W3/2))
            exist = self.exist_head(x)      # (B, max_num_lanes)
            output['pred_exist'] = exist

        return output

    def loss(self,
             preds_dicts,
             img_metas,
             gt_semantic_seg,
             lane_exist=None,
             ):
        pred_seg = preds_dicts['pred_seg']      # (B, seg_num_classes, H, W)
        gt_semantic_seg = gt_semantic_seg.squeeze(dim=1).long()     # (B, H, W)

        num_total_pos = (gt_semantic_seg > 0).sum()
        num_total_neg = (gt_semantic_seg == 0).sum()
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                pred_seg.new_tensor([cls_avg_factor]))

        loss_seg = self.loss_seg(
            pred_seg,
            gt_semantic_seg,
            avg_factor=cls_avg_factor
        )

        loss_dict = dict()
        loss_dict['loss_seg'] = loss_seg

        if self.with_exist:
            pred_exist = preds_dicts['pred_exist']      # (B, num_lanes)
            gt_exist = torch.stack(lane_exist, dim=0)   # (B, num_lanes)

            loss_exist = self.loss_exist(
                pred_exist,
                gt_exist
            )
            loss_dict['loss_exist'] = loss_exist

        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_lanes(self, preds_dicts, img_metas, cfg=None):
        '''
        Convert model output to lanes.
        Args:
            preds_dicts = dict {
                'pred_seg': (B, seg_num_classes, H, W)
                'pred_exist': (B, max_num_lanes)
            }
        Returns:
            results_list: Lane results of each image. The outer list corresponds to each image. The inner list
                corresponds to each lane.
                List[List[lane0, lane1, ...], ...]
        '''
        pred_seg = preds_dicts['pred_seg']      # (B, seg_num_classes, H, W)
        if self.use_sigmoid_seg:
            pred_seg = pred_seg.sigmoid()
        else:
            pred_seg = F.softmax(pred_seg, dim=1)[:, 1:]       # (B, max_num_lanes, H, W)
        pred_seg = pred_seg.detach().clone()

        if preds_dicts.get('pred_exist', None) is not None:
            pred_exist = preds_dicts['pred_exist']      # (B, max_num_lanes)
            assert self.use_sigmoid_exist is True
            pred_exist = pred_exist.sigmoid()       # (B, max_num_lanes)
            pred_exist = pred_exist.detach().clone()
            exists = pred_exist > self.test_cfg['exist_score_thr']
        else:
            exists = [None for _ in pred_seg]

        results_list = []
        for seg, exist, img_meta in zip(pred_seg, exists, img_metas):
            # seg: (max_num_lanes, H, W)
            # exist: (max_num_lanes, )   bool
            pred = self.predictions_to_pred(seg, exist, img_meta)
            results_list.append(pred)
        return results_list

    # def predictions_to_pred(self, probmaps, exists, img_meta):
    #     """
    #     :param probmaps: (max_num_lanes, H, W)
    #     :param exists: (max_num_lanes, )   bool
    #     :param img_meta:
    #     :return:
    #         lanes: List[lane0, lane1, ...]
    #     """
    #     ori_height, ori_width = img_meta['ori_shape'][:2]
    #     crop = img_meta.get('crop', (0, 0, 0, 0))  # (x_min, y_min, x_max, y_max)
    #     y_min = crop[1]
    #
    #     lanes = []
    #     if exists is None:
    #         exists = [True for _ in probmaps]
    #     for probmap, exist in zip(probmaps, exists):
    #         if not exist:
    #             continue
    #         probmap = probmap.cpu().numpy()
    #         probmap = cv2.blur(probmap, (9, 9), borderType=cv2.BORDER_REPLICATE)
    #
    #         coord = []
    #         for y in self.prior_ys:
    #             proj_y = round((self.img_h - 1) * y.item())
    #             line = probmap[proj_y]
    #             if np.max(line) < self.test_cfg['seg_score_thr']:
    #                 continue
    #             value = np.argmax(line)
    #             x = value * ori_width / self.img_w
    #             if x >= 0:
    #                 y = y.item() * ((ori_height - 1) - y_min) + y_min
    #                 coord.append([x, y])
    #         if len(coord) < self.test_cfg['min_num_lanes']:
    #             continue
    #
    #         coord = np.array(coord)
    #         coord[:, 0] /= ori_width
    #         coord[:, 1] /= ori_height
    #         lanes.append(Lane(coord))
    #
    #     return lanes

    def predictions_to_pred(self, probmaps, exists, img_meta):
        """
        :param probmaps: (max_num_lanes, H, W)
        :param exists: (max_num_lanes, )   bool
        :param img_meta:
        :return:
            lanes: List[lane0, lane1, ...]
        """
        ori_height, ori_width = img_meta['ori_shape'][:2]
        crop = img_meta.get('crop', (0, 0, 0, 0))  # (x_min, y_min, x_max, y_max)
        cut_height = crop[1]
        ori_cropped_height = ori_height - cut_height

        lanes = []
        if exists is None:
            exists = [True for _ in probmaps]
        for probmap, exist in zip(probmaps, exists):
            if not exist:
                continue
            probmap = probmap.cpu().numpy()
            probmap = cv2.blur(probmap, (9, 9), borderType=cv2.BORDER_REPLICATE)

            coord = []
            for y in self.test_cfg['sample_y']:
                proj_y = round((y - cut_height) * self.img_h / ori_cropped_height)
                line = probmap[proj_y]
                if np.max(line) < self.test_cfg['seg_score_thr']:
                    continue
                value = np.argmax(line)
                x = value * ori_width / self.img_w
                if x > 0:
                    coord.append([x, y])

            if len(coord) < self.test_cfg['min_num_lanes']:
                continue

            coord = np.array(coord)
            coord = np.flip(coord, axis=0)
            coord[:, 0] /= ori_width
            coord[:, 1] /= ori_height
            lanes.append(Lane(coord))

        return lanes
