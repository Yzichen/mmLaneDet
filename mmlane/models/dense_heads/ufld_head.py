import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from mmdet.core import reduce_mean
from mmlane.models import build_loss
from mmcv.cnn import ConvModule
from ..builder import HEADS
from .base_dense_head import BaseDenseHead
from mmcv.runner import force_fp32
import scipy
from mmlane.core import Lane


@HEADS.register_module
class AuxSegHead(nn.Module):
    def __init__(self,
                 base_in_channel=128,
                 seg_out_channels=6,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')
                 ):
        super(AuxSegHead, self).__init__()
        self.aux_header2 = nn.Sequential(
            ConvModule(in_channels=base_in_channel,
                       out_channels=128,
                       kernel_size=3,
                       padding=1,
                       conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg
                       ),
            ConvModule(in_channels=128, out_channels=128, kernel_size=3, padding=1, conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(in_channels=128, out_channels=128, kernel_size=3, padding=1, conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(in_channels=128, out_channels=128, kernel_size=3, padding=1, conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
        )

        self.aux_header3 = nn.Sequential(
            ConvModule(in_channels=base_in_channel*2,
                       out_channels=128,
                       kernel_size=3,
                       padding=1,
                       conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg
                       ),
            ConvModule(in_channels=128, out_channels=128, kernel_size=3, padding=1, conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(in_channels=128, out_channels=128, kernel_size=3, padding=1, conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
        )

        self.aux_header4 = nn.Sequential(
            ConvModule(in_channels=base_in_channel*4,
                       out_channels=128,
                       kernel_size=3,
                       padding=1,
                       conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg
                       ),
            ConvModule(in_channels=128, out_channels=128, kernel_size=3, padding=1, conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
        )

        self.aux_combine = nn.Sequential(
            ConvModule(384, 256, kernel_size=3, padding=2, dilation=2, conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(256, 128, kernel_size=3, padding=2, dilation=2, conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(128, 128, kernel_size=3, padding=2, dilation=2, conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(128, 128, kernel_size=3, padding=4, dilation=4, conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            nn.Conv2d(128, seg_out_channels, kernel_size=1)
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x2, x3, fea = x     # (B, C3, H3, W3), (B, C4, H4, W4), (B, C5, H5, W5)
        x2 = self.aux_header2(x2)       # (B, 128, H3, W3)

        x3 = self.aux_header3(x3)       # (B, 128, H4, W4)
        x3 = torch.nn.functional.interpolate(x3, scale_factor=2, mode='bilinear')   # (B, 128, H3, W3)

        x4 = self.aux_header4(fea)      # (B, 128, H5, W5)
        x4 = torch.nn.functional.interpolate(x4, scale_factor=4, mode='bilinear')   # (B, 128, H3, W3)

        aux_seg = torch.cat([x2, x3, x4], dim=1)    # (B, 128*3=384, H3, W3)
        aux_seg = self.aux_combine(aux_seg)     # (B, n_lanes+1, H3, W3)

        return aux_seg


@HEADS.register_module
class UFLDHead(BaseDenseHead):
    def __init__(self,
                 in_channels=512,
                 row_anchor_num=56,
                 grid_num=100,
                 max_num_lanes=6,
                 with_seg=True,
                 feature_map_shape=(9, 25),
                 pool_out_channel=8,
                 loss_seg=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     class_weight=1.0,
                     bg_cls_weight=0.4,
                     loss_weight=1.0,
                     reduction='mean',
                 ),
                 loss_cls=dict(
                     type='SoftmaxFocalloss',
                     use_sigmoid=False,
                     gamma=2.0,
                     alpha=1.0,
                     reduction='mean',
                     loss_weight=1.0
                 ),
                 sync_cls_avg_factor=False,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 ):
        super(UFLDHead, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.row_anchor_num = row_anchor_num
        self.grid_num = grid_num
        self.max_num_lanes = max_num_lanes

        h, w = feature_map_shape
        self.pool = nn.Conv2d(in_channels=in_channels, out_channels=pool_out_channel, kernel_size=1)
        self.in_dim = h * w * pool_out_channel
        mid_dim = 2048
        total_dim = row_anchor_num * (grid_num + 1) * max_num_lanes
        self.cls = nn.Sequential(
            nn.Linear(self.in_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, total_dim)
        )

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = grid_num
        else:
            self.cls_out_channels = grid_num + 1

        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor

        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is UFLDHead):
            assert isinstance(class_weight, float), 'Expected ' \
                                                    'class_weight to have type float. Found ' \
                                                    f'{type(class_weight)}.'
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                                                     'bg_cls_weight to have type float. Found ' \
                                                     f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(grid_num + 1) * class_weight
            # set background class as the last indice
            class_weight[-1] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')

            self.bg_cls_weight = bg_cls_weight

        self.loss_cls = build_loss(loss_cls)

        self.with_seg = with_seg
        self.seg_bg_cls_weight = 0
        if self.with_seg:
            seg_num_classes = max_num_lanes
            self.use_sigmoid_seg = loss_seg.get('use_sigmoid', False)
            if self.use_sigmoid_seg:
                self.seg_out_channels = seg_num_classes
            else:
                self.seg_out_channels = seg_num_classes + 1

            self.aux_seg_head = AuxSegHead(base_in_channel=in_channels//4, seg_out_channels=self.seg_out_channels)

            class_weight = loss_seg.get('class_weight', None)
            if class_weight is not None:
                bg_cls_weight = loss_seg.get('bg_cls_weight', class_weight)
                class_weight = torch.ones(seg_num_classes + 1) * class_weight
                assert isinstance(bg_cls_weight, float), 'Expected ' \
                                                         'bg_cls_weight to have type float. Found ' \
                                                         f'{type(bg_cls_weight)}.'
                class_weight[0] = bg_cls_weight  # 这里0是bg, 而mmdet系列是num_class是bg.
                loss_seg.update({'class_weight': class_weight})
                if 'bg_cls_weight' in loss_seg:
                    loss_seg.pop('bg_cls_weight')
                self.seg_bg_cls_weight = bg_cls_weight
            self.loss_seg = build_loss(loss_seg)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights()

    def init_weights(self):
        for m in self.cls.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             preds_dicts,
             img_metas,
             row_anchor_label,
             gt_semantic_seg=None,
             ):
        """Compute losses of the head.
        Args:
            preds_dicts: dict{
                pred_cls: (B, num_grids+1, N_row_anchors, N_lanes)
                pred_seg: (B, N_lanes+1, H3, W3)
            }
            img_metas:
            gt_semantic_seg (Tensor): (B, 1, H, W)
            row_anchor_label (Tensor): (B, N_row_anchors, N_lanes)
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        loss_dict = dict()

        cls_score = preds_dicts['pred_cls']     # (B, num_grids+1, N_row_anchors, N_lanes)
        loss_cls = self.loss_cls(
            cls_score,
            row_anchor_label.long(),
        )
        loss_dict['loss_cls'] = loss_cls

        if self.with_seg:
            pred_seg = preds_dicts['pred_seg']      # (B, seg_num_classes=N_lanes+1, H3, W3)
            gt_semantic_seg = F.interpolate(gt_semantic_seg, scale_factor=1/8, mode='nearest')
            gt_semantic_seg = gt_semantic_seg.squeeze(dim=1).long()     # (B, H3, W3)
            num_total_pos = (gt_semantic_seg > 0).sum()
            num_total_neg = (gt_semantic_seg == 0).sum()
            cls_avg_factor = num_total_pos * 1.0 + \
                num_total_neg * self.seg_bg_cls_weight

            if self.sync_cls_avg_factor:
                cls_avg_factor = reduce_mean(
                    pred_seg.new_tensor([cls_avg_factor]))

            loss_seg = self.loss_seg(
                pred_seg,
                gt_semantic_seg,
                avg_factor=cls_avg_factor
            )
            loss_dict['loss_seg'] = loss_seg

        return loss_dict

    def forward(self, x, **kwargs):
        '''
        Args:
            x: input features (list[Tensor])
            kwargs:
                dict {
                    'img': (B, 3, img_H, img_W),
                    'seg': (B, img_H, img_W),
                    'meta': List[dict0, dict1, ...]
                }
        Return:
            output:
                pred_cls: (B, num_grids+1, N_row_anchors, N_lanes)
                pred_seg: (B, N_lanes+1, H3, W3)
        '''
        fea = x[-1]
        fea = self.pool(fea).view(-1, self.in_dim)     # (B, 8, H5=9, W5=25) --> (B, 1800)
        # (B, 1800) --> (B, 2048) --> c
        pred_cls = self.cls(fea).view(-1, self.grid_num+1, self.row_anchor_num, self.max_num_lanes)
        output = {'pred_cls': pred_cls}

        if self.with_seg:
            pred_seg = self.aux_seg_head(x)     # (B, n_lanes+1, H3, W3)
            output['pred_seg'] = pred_seg

        return output

    @force_fp32(apply_to=('preds_dicts'))
    def get_lanes(self, preds_dicts, img_metas, cfg=None):
        '''
        Convert model output to lanes.
        Args:
            preds_dicts = dict {
                pred_cls: (B, num_grids+1, N_row_anchors, N_lanes)
                pred_seg: (B, N_lanes+1, H3, W3)
            }
        Returns:
            results_list: Lane results of each image. The outer list corresponds to each image. The inner list
                corresponds to each lane.
                List[List[lane0, lane1, ...], ...]
        '''
        pred_cls = preds_dicts['pred_cls']      # (B, num_grids+1, N_row_anchors, N_lanes)
        pred_cls = pred_cls.detach().clone()

        localization_type = self.test_cfg['localization_type']

        results_list = []
        for cls, img_meta in zip(pred_cls, img_metas):
            cls = cls.cpu().numpy()     # (num_grids+1, N_row_anchors, N_lanes)

            if localization_type == 'abs':
                row_anchor_pred = np.argmax(cls, axis=0)    # (N_row_anchors, N_lanes)
                row_anchor_pred[row_anchor_pred == self.grid_num] = -1
            elif localization_type == 'rel':
                prob = scipy.special.softmax(cls[:-1, :, :], axis=0)    # (num_grids, N_row_anchors, N_lanes)
                grid_bins = np.arange(self.grid_num).reshape(-1, 1, 1)     # (num_grids, 1, 1)
                row_anchor_pred = np.sum(prob * grid_bins, axis=0)      # (N_row_anchors, N_lanes)
                max_idx = np.argmax(cls, axis=0)
                row_anchor_pred[max_idx == self.grid_num] = -1
            else:
                raise NotImplementedError

            pred = self.predictions_to_pred(row_anchor_pred, img_meta)
            results_list.append(pred)
        return results_list

    def predictions_to_pred(self, row_anchor_preds, img_meta):
        """
        :param row_anchor_pred: (N_row_anchors, N_lanes)
        :param img_meta:
        :return:
            lanes: List[lane0, lane1, ...]
        """
        ori_height, ori_width = img_meta['ori_shape'][:2]
        img_h, img_w = img_meta['img_shape'][:2]
        crop = img_meta.get('crop', (0, 0, 0, 0))  # (x_min, y_min, x_max, y_max)
        y_min = crop[1]
        ratio_y = (ori_height - y_min) / img_h

        sample_y = self.test_cfg['sample_y']
        lanes = []

        for lane_idx in range(row_anchor_preds.shape[1]):
            cur_row_anchor_pred = row_anchor_preds[:, lane_idx]      # (N_row_anchors, )
            if sum(cur_row_anchor_pred != -1) <= 2:
                continue

            coord = []
            for row_id in range(len(cur_row_anchor_pred)):
                if cur_row_anchor_pred[row_id] < 0:
                    continue
                x = cur_row_anchor_pred[row_id] * (ori_width - 1) / (self.grid_num - 1)
                y = sample_y[row_id] * ratio_y + y_min
                coord.append([x, y])

            coord = np.array(coord)
            coord[:, 0] /= ori_width
            coord[:, 1] /= ori_height

            lanes.append(Lane(coord))

        return lanes


# @HEADS.register_module
# class UFLDHeadPlus(BaseDenseHead):
#     def __init__(self,
#                  in_channels=512,
#                  row_anchor_num=56,
#                  grid_num=100,
#                  max_num_lanes=6,
#                  with_seg=True,
#                  feature_map_shape=(9, 25),
#                  pool_out_channel=8,
#                  loss_seg=dict(
#                      type='CrossEntropyLoss',
#                      use_sigmoid=False,
#                      class_weight=1.0,
#                      bg_cls_weight=0.4,
#                      loss_weight=1.0,
#                      reduction='mean',
#                  ),
#                  loss_location=dict(
#                      type='SmoothL1Loss',
#                      beta=1.0,
#                      reduction='mean',
#                      loss_weight=1.0
#                  ),
#                  loss_exist=dict(
#                      type='CrossEntropyLoss',
#                      use_sigmoid=False,
#                      reduction='mean',
#                      loss_weight=1.0
#                  ),
#                  sync_cls_avg_factor=False,
#                  train_cfg=None,
#                  test_cfg=None,
#                  init_cfg=None,
#                  ):
#         super(UFLDHeadPlus, self).__init__(init_cfg)
#         self.in_channels = in_channels
#         self.row_anchor_num = row_anchor_num
#         self.grid_num = grid_num
#         self.max_num_lanes = max_num_lanes
#
#         h, w = feature_map_shape
#         self.pool = nn.Conv2d(in_channels=in_channels, out_channels=pool_out_channel, kernel_size=1)
#         self.in_dim = h * w * pool_out_channel
#         mid_dim = 2048
#         location_total_dim = row_anchor_num * grid_num * max_num_lanes
#         self.shared_branch = nn.Sequential(
#             nn.Linear(self.in_dim, mid_dim),
#             nn.ReLU(),
#         )
#         # 位置分类估计.
#         self.location_cls = nn.Linear(mid_dim, location_total_dim)
#
#         # 用于预测该row anchor处是否存在lane.
#         use_sigmoid_cls = loss_exist.get('use_sigmoid', False)
#         if use_sigmoid_cls:
#             self.num_exist_out = 1
#         else:
#             self.num_exist_out = 2
#         exist_total_dim = row_anchor_num * self.num_exist_out * max_num_lanes
#         self.exist_cls = nn.Linear(mid_dim, exist_total_dim)
#
#         self.bg_cls_weight = 0
#         self.sync_cls_avg_factor = sync_cls_avg_factor
#
#         self.loss_exist = build_loss(loss_exist)
#         self.loss_location = build_loss(loss_location)
#
#         self.with_seg = with_seg
#         self.seg_bg_cls_weight = 0
#         if self.with_seg:
#             seg_num_classes = max_num_lanes
#             self.use_sigmoid_seg = loss_seg.get('use_sigmoid', False)
#             if self.use_sigmoid_seg:
#                 self.seg_out_channels = seg_num_classes
#             else:
#                 self.seg_out_channels = seg_num_classes + 1
#
#             self.aux_seg_head = AuxSegHead(base_in_channel=in_channels//4, seg_out_channels=self.seg_out_channels)
#
#             class_weight = loss_seg.get('class_weight', None)
#             if class_weight is not None:
#                 bg_cls_weight = loss_seg.get('bg_cls_weight', class_weight)
#                 class_weight = torch.ones(seg_num_classes + 1) * class_weight
#                 assert isinstance(bg_cls_weight, float), 'Expected ' \
#                                                          'bg_cls_weight to have type float. Found ' \
#                                                          f'{type(bg_cls_weight)}.'
#                 class_weight[0] = bg_cls_weight  # 这里0是bg, 而mmdet系列是num_class是bg.
#                 loss_seg.update({'class_weight': class_weight})
#                 if 'bg_cls_weight' in loss_seg:
#                     loss_seg.pop('bg_cls_weight')
#                 self.seg_bg_cls_weight = bg_cls_weight
#             self.loss_seg = build_loss(loss_seg)
#
#         self.train_cfg = train_cfg
#         self.test_cfg = test_cfg
#
#     @force_fp32(apply_to=('preds_dicts'))
#     def loss(self,
#              preds_dicts,
#              img_metas,
#              row_anchor_label,
#              gt_semantic_seg=None,
#              ):
#         """Compute losses of the head.
#         Args:
#             preds_dicts: dict{
#                 pred_cls: (B, num_grids, N_row_anchors, N_lanes)
#                 pred_seg: (B, N_lanes+1, H3, W3)
#             }
#             img_metas:
#             gt_semantic_seg (Tensor): (B, 1, H, W)
#             row_anchor_label (Tensor): (B, N_row_anchors, N_lanes)
#         Returns:
#             dict[str, Tensor]: A dictionary of loss components.
#         """
#         loss_dict = dict()
#
#         # 计算location loss, 使用smooth-l1 loss.
#         location_cls = preds_dicts['location_cls']     # (B, num_bins, N_row_anchors, N_lanes)
#         batch_size = location_cls.shape[0]
#         bins = torch.arange(0, self.grid_num, step=1, device=location_cls.device)
#         # (num_bins, ) --> (1, num_bins, 1, 1) --> (B, num_bins, N_row_anchors, N_lanes)
#         bins = bins.view(1, self.grid_num, 1, 1).repeat(batch_size, 1, self.row_anchor_num,
#                                                         self.max_num_lanes)
#         location_cls = F.softmax(location_cls, dim=1)
#         # (B, num_bins, N_row_anchors, N_lanes) * (B, num_bins, N_row_anchors, N_lanes)
#         # --> (B, N_row_anchors, N_lanes)
#         pred_location = torch.sum(location_cls * bins, dim=1)
#         location_mask = row_anchor_label >= 0   # (B, N_row_anchors, N_lanes)
#
#         loss_location = self.loss_location(
#             pred_location,
#             row_anchor_label,
#             weight=location_mask,
#             avg_factor=location_mask.sum()
#         )
#         loss_dict['loss_location'] = loss_location
#
#         # 计算exist loss, 使用cross-entropy loss.
#         pred_exist = preds_dicts['exist_cls']    # (B, 2, N_row_anchors, N_lanes)
#         row_anchor_exist_label = location_mask.long()      # (B, N_row_anchors, N_lanes),  1：说明存在， 0：不存在.
#
#         loss_exist = self.loss_exist(
#             pred_exist,
#             row_anchor_exist_label,
#         )
#         loss_dict['loss_exist'] = loss_exist
#
#         if self.with_seg:
#             pred_seg = preds_dicts['pred_seg']      # (B, seg_num_classes=N_lanes+1, H3, W3)
#             gt_semantic_seg = F.interpolate(gt_semantic_seg, scale_factor=1/8, mode='nearest')
#             gt_semantic_seg = gt_semantic_seg.squeeze(dim=1).long()     # (B, H3, W3)
#             num_total_pos = (gt_semantic_seg > 0).sum()
#             num_total_neg = (gt_semantic_seg == 0).sum()
#             cls_avg_factor = num_total_pos * 1.0 + \
#                 num_total_neg * self.seg_bg_cls_weight
#
#             if self.sync_cls_avg_factor:
#                 cls_avg_factor = reduce_mean(
#                     pred_seg.new_tensor([cls_avg_factor]))
#
#             loss_seg = self.loss_seg(
#                 pred_seg,
#                 gt_semantic_seg,
#                 avg_factor=cls_avg_factor
#             )
#             loss_dict['loss_seg'] = loss_seg
#
#         return loss_dict
#
#     def forward(self, x, **kwargs):
#         '''
#         Args:
#             x: input features (list[Tensor])
#             kwargs:
#                 dict {
#                     'img': (B, 3, img_H, img_W),
#                     'seg': (B, img_H, img_W),
#                     'meta': List[dict0, dict1, ...]
#                 }
#         Return:
#             output:
#                 pred_cls: (B, num_grids+1, N_row_anchors, N_lanes)
#                 pred_seg: (B, N_lanes+1, H3, W3)
#         '''
#         fea = x[-1]
#         fea = self.pool(fea).view(-1, self.in_dim)     # (B, 8, H5=9, W5=25) --> (B, 1800)
#         fea = self.shared_branch(fea)   # (B, 1800) --> (B, 2048)
#         # (B, 2048) --> (B, grid_num, N_row_anchors, N_lanes)
#         location_cls = self.location_cls(fea).view(-1, self.grid_num, self.row_anchor_num, self.max_num_lanes)
#         # (B, 2048) --> (B, 2, N_row_anchors, N_lanes)
#         exist_cls = self.exist_cls(fea).view(-1, self.num_exist_out, self.row_anchor_num, self.max_num_lanes)
#         output = {'location_cls': location_cls,
#                   'exist_cls': exist_cls}
#
#         if self.with_seg:
#             pred_seg = self.aux_seg_head(x)     # (B, n_lanes+1, H3, W3)
#             output['pred_seg'] = pred_seg
#
#         return output
#
#     @force_fp32(apply_to=('preds_dicts'))
#     def get_lanes(self, preds_dicts, img_metas, cfg=None):
#         '''
#         Convert model output to lanes.
#         Args:
#             preds_dicts = dict {
#                 location_cls: (B, grid_num, N_row_anchors, N_lanes)
#                 exist_cls: (B, 2, N_row_anchors, N_lanes)
#             }
#         Returns:
#             results_list: Lane results of each image. The outer list corresponds to each image. The inner list
#                 corresponds to each lane.
#                 List[List[lane0, lane1, ...], ...]
#         '''
#         def get_range(exists):
#             """
#             Args:
#                 exists: (N_row_anchors, N_lanes)
#             Returns:
#                 lane_ranges: List[[min_idx, max_idx], [min_idx, max_idx], ...]
#             """
#             max_rows, max_lanes = exists.shape
#             lane_ranges = []
#             for idx in range(max_lanes):
#                 cur_exists = exists[:, idx]     # (N_row_anchors, )
#                 min_idx = max_idx = None
#                 for row_idx, valid in enumerate(cur_exists):
#                     if valid:
#                         min_idx = row_idx - 1
#                         break
#                 for row_idx, valid in enumerate(cur_exists[::-1]):
#                     if valid:
#                         max_idx = len(cur_exists) - row_idx
#                         break
#                 if max_idx is not None:
#                     max_idx = min(max_rows - 1, max_idx)
#                 if min_idx is not None:
#                     min_idx = max(0, min_idx)
#                 lane_ranges.append([min_idx, max_idx])
#             return lane_ranges
#
#         batch_location_cls = preds_dicts['location_cls']      # (B, num_grids, N_row_anchors, N_lanes)
#         batch_exist_cls = preds_dicts['exist_cls']    # (B, 2, N_row_anchors, N_lanes)
#         batch_location_cls = batch_location_cls.detach().clone()
#         batch_exist_cls = batch_exist_cls.detach().clone()
#
#         results_list = []
#         for location_cls, exist_cls, img_meta in zip(batch_location_cls, batch_exist_cls, img_metas):
#             location_cls = location_cls.cpu().numpy()     # (num_grids, N_row_anchors, N_lanes)
#             prob = scipy.special.softmax(location_cls, axis=0)    # (num_grids, N_row_anchors, N_lanes)
#             grid_bins = np.arange(self.grid_num).reshape(-1, 1, 1)     # (num_grids, 1, 1)
#             row_anchor_pred = np.sum(prob * grid_bins, axis=0)      # (N_row_anchors, N_lanes)
#
#             # 获得预测车道的范围
#             exist_cls = exist_cls.cpu().numpy()     # (2, N_row_anchors, N_lanes)
#             exists = np.argmax(exist_cls, axis=0)      # (N_row_anchors, N_lanes)  # 1：说明存在， 0：不存在.
#             lane_ranges = get_range(exists)      # List[[min_y, max_y], [min_y, max_y], ...]
#
#             pred = self.predictions_to_pred(row_anchor_pred, lane_ranges, img_meta)
#             results_list.append(pred)
#         return results_list
#
#     def predictions_to_pred(self, row_anchor_preds, lane_ranges, img_meta):
#         """
#         :param row_anchor_preds: (N_row_anchors, N_lanes)
#         :param lane_ranges: List[[min_y, max_y], [min_y, max_y], ...]
#         :param img_meta:
#         :return:
#             lanes: List[lane0, lane1, ...]
#         """
#         ori_height, ori_width = img_meta['ori_shape'][:2]
#         img_h, img_w = img_meta['img_shape'][:2]
#         crop = img_meta.get('crop', (0, 0, 0, 0))  # (x_min, y_min, x_max, y_max)
#         y_min = crop[1]
#         ratio_y = (ori_height - y_min) / img_h
#
#         sample_y = self.test_cfg['sample_y']
#         sample_y = np.array(sample_y, dtype=np.float32)
#         lanes = []
#
#         for lane_idx in range(row_anchor_preds.shape[1]):
#             min_y, max_y = lane_ranges[lane_idx][0], lane_ranges[lane_idx][1]
#             if min_y is None or max_y is None or (max_y - min_y) <= 2:
#                 continue
#
#             # (N_valid_anchors,  )
#             vaild_sample_y = sample_y[min_y:max_y]
#             valid_row_anchor_pred = row_anchor_preds[min_y:max_y, lane_idx]
#
#             # 缩放到原始图片的比例
#             y = vaild_sample_y * ratio_y + y_min
#             x = valid_row_anchor_pred * (ori_width - 1) / (self.grid_num - 1)
#
#             coord = np.stack((x, y), axis=1)
#             coord[:, 0] /= ori_width
#             coord[:, 1] /= ori_height
#
#             lanes.append(Lane(coord))
#
#         return lanes


# exist分类是依靠location cls的结果.
@HEADS.register_module
class UFLDHeadPlus(BaseDenseHead):
    def __init__(self,
                 in_channels=512,
                 row_anchor_num=56,
                 grid_num=100,
                 max_num_lanes=6,
                 with_seg=True,
                 feature_map_shape=(9, 25),
                 pool_out_channel=8,
                 loss_seg=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     class_weight=1.0,
                     bg_cls_weight=0.4,
                     loss_weight=1.0,
                     reduction='mean',
                 ),
                 loss_location=dict(
                     type='SmoothL1Loss',
                     beta=1.0,
                     reduction='mean',
                     loss_weight=1.0
                 ),
                 loss_exist=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     reduction='mean',
                     loss_weight=1.0
                 ),
                 sync_cls_avg_factor=False,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 ):
        super(UFLDHeadPlus, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.row_anchor_num = row_anchor_num
        self.grid_num = grid_num
        self.max_num_lanes = max_num_lanes

        h, w = feature_map_shape
        self.pool = nn.Conv2d(in_channels=in_channels, out_channels=pool_out_channel, kernel_size=1)
        self.in_dim = h * w * pool_out_channel
        mid_dim = 2048
        location_total_dim = row_anchor_num * grid_num * max_num_lanes
        self.shared_branch = nn.Sequential(
            nn.Linear(self.in_dim, mid_dim),
            nn.ReLU(),
        )
        # 位置分类估计.
        self.location_cls = nn.Linear(mid_dim, location_total_dim)

        # 用于预测该row anchor处是否存在lane.
        use_sigmoid_cls = loss_exist.get('use_sigmoid', False)
        if use_sigmoid_cls:
            self.num_exist_out = 1
        else:
            self.num_exist_out = 2
        exist_total_dim = row_anchor_num * self.num_exist_out * max_num_lanes
        # self.exist_cls = nn.Linear(mid_dim, exist_total_dim)
        self.exist_cls = nn.Conv2d(self.grid_num, self.num_exist_out, kernel_size=1, stride=1)

        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor

        self.loss_exist = build_loss(loss_exist)
        self.loss_location = build_loss(loss_location)

        self.with_seg = with_seg
        self.seg_bg_cls_weight = 0
        if self.with_seg:
            seg_num_classes = max_num_lanes
            self.use_sigmoid_seg = loss_seg.get('use_sigmoid', False)
            if self.use_sigmoid_seg:
                self.seg_out_channels = seg_num_classes
            else:
                self.seg_out_channels = seg_num_classes + 1

            self.aux_seg_head = AuxSegHead(base_in_channel=in_channels//4, seg_out_channels=self.seg_out_channels)

            class_weight = loss_seg.get('class_weight', None)
            if class_weight is not None:
                bg_cls_weight = loss_seg.get('bg_cls_weight', class_weight)
                class_weight = torch.ones(seg_num_classes + 1) * class_weight
                assert isinstance(bg_cls_weight, float), 'Expected ' \
                                                         'bg_cls_weight to have type float. Found ' \
                                                         f'{type(bg_cls_weight)}.'
                class_weight[0] = bg_cls_weight  # 这里0是bg, 而mmdet系列是num_class是bg.
                loss_seg.update({'class_weight': class_weight})
                if 'bg_cls_weight' in loss_seg:
                    loss_seg.pop('bg_cls_weight')
                self.seg_bg_cls_weight = bg_cls_weight
            self.loss_seg = build_loss(loss_seg)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             preds_dicts,
             img_metas,
             row_anchor_label,
             gt_semantic_seg=None,
             ):
        """Compute losses of the head.
        Args:
            preds_dicts: dict{
                pred_cls: (B, num_grids, N_row_anchors, N_lanes)
                pred_seg: (B, N_lanes+1, H3, W3)
            }
            img_metas:
            gt_semantic_seg (Tensor): (B, 1, H, W)
            row_anchor_label (Tensor): (B, N_row_anchors, N_lanes)
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        loss_dict = dict()

        # 计算location loss, 使用smooth-l1 loss.
        location_cls = preds_dicts['location_cls']     # (B, num_bins, N_row_anchors, N_lanes)
        batch_size = location_cls.shape[0]
        bins = torch.arange(0, self.grid_num, step=1, device=location_cls.device)
        # (num_bins, ) --> (1, num_bins, 1, 1) --> (B, num_bins, N_row_anchors, N_lanes)
        bins = bins.view(1, self.grid_num, 1, 1).repeat(batch_size, 1, self.row_anchor_num,
                                                        self.max_num_lanes)
        location_cls = F.softmax(location_cls, dim=1)
        # (B, num_bins, N_row_anchors, N_lanes) * (B, num_bins, N_row_anchors, N_lanes)
        # --> (B, N_row_anchors, N_lanes)
        pred_location = torch.sum(location_cls * bins, dim=1)
        location_mask = row_anchor_label >= 0   # (B, N_row_anchors, N_lanes)

        loss_location = self.loss_location(
            pred_location,
            row_anchor_label,
            weight=location_mask,
            avg_factor=location_mask.sum()
        )
        loss_dict['loss_location'] = loss_location

        # 计算exist loss, 使用cross-entropy loss.
        pred_exist = preds_dicts['exist_cls']    # (B, 2, N_row_anchors, N_lanes)
        row_anchor_exist_label = location_mask.long()      # (B, N_row_anchors, N_lanes),  1：说明存在， 0：不存在.

        loss_exist = self.loss_exist(
            pred_exist,
            row_anchor_exist_label,
        )
        loss_dict['loss_exist'] = loss_exist

        if self.with_seg:
            pred_seg = preds_dicts['pred_seg']      # (B, seg_num_classes=N_lanes+1, H3, W3)
            gt_semantic_seg = F.interpolate(gt_semantic_seg, scale_factor=1/8, mode='nearest')
            gt_semantic_seg = gt_semantic_seg.squeeze(dim=1).long()     # (B, H3, W3)
            num_total_pos = (gt_semantic_seg > 0).sum()
            num_total_neg = (gt_semantic_seg == 0).sum()
            cls_avg_factor = num_total_pos * 1.0 + \
                num_total_neg * self.seg_bg_cls_weight

            if self.sync_cls_avg_factor:
                cls_avg_factor = reduce_mean(
                    pred_seg.new_tensor([cls_avg_factor]))

            loss_seg = self.loss_seg(
                pred_seg,
                gt_semantic_seg,
                avg_factor=cls_avg_factor
            )
            loss_dict['loss_seg'] = loss_seg

        return loss_dict

    def forward(self, x, **kwargs):
        '''
        Args:
            x: input features (list[Tensor])
            kwargs:
                dict {
                    'img': (B, 3, img_H, img_W),
                    'seg': (B, img_H, img_W),
                    'meta': List[dict0, dict1, ...]
                }
        Return:
            output:
                pred_cls: (B, num_grids+1, N_row_anchors, N_lanes)
                pred_seg: (B, N_lanes+1, H3, W3)
        '''
        fea = x[-1]
        fea = self.pool(fea).view(-1, self.in_dim)     # (B, 8, H5=9, W5=25) --> (B, 1800)
        fea = self.shared_branch(fea)   # (B, 1800) --> (B, 2048)
        # (B, 2048) --> (B, grid_num, N_row_anchors, N_lanes)
        location_cls = self.location_cls(fea).view(-1, self.grid_num, self.row_anchor_num, self.max_num_lanes)

        # (B, 2048) --> (B, 2, N_row_anchors, N_lanes)
        # exist_cls = self.exist_cls(fea).view(-1, self.num_exist_out, self.row_anchor_num, self.max_num_lanes)

        # (B, grid_num, N_row_anchors, N_lanes) --> (B, 2, N_row_anchors, N_lanes)
        exist_cls = self.exist_cls(location_cls).view(-1, self.num_exist_out, self.row_anchor_num, self.max_num_lanes)
        output = {'location_cls': location_cls,
                  'exist_cls': exist_cls}

        if self.with_seg:
            pred_seg = self.aux_seg_head(x)     # (B, n_lanes+1, H3, W3)
            output['pred_seg'] = pred_seg

        return output

    @force_fp32(apply_to=('preds_dicts'))
    def get_lanes(self, preds_dicts, img_metas, cfg=None):
        '''
        Convert model output to lanes.
        Args:
            preds_dicts = dict {
                location_cls: (B, grid_num, N_row_anchors, N_lanes)
                exist_cls: (B, 2, N_row_anchors, N_lanes)
            }
        Returns:
            results_list: Lane results of each image. The outer list corresponds to each image. The inner list
                corresponds to each lane.
                List[List[lane0, lane1, ...], ...]
        '''
        def get_range(exists):
            """
            Args:
                exists: (N_row_anchors, N_lanes)
            Returns:
                lane_ranges: List[[min_idx, max_idx], [min_idx, max_idx], ...]
            """
            max_rows, max_lanes = exists.shape
            lane_ranges = []
            for idx in range(max_lanes):
                cur_exists = exists[:, idx]     # (N_row_anchors, )
                min_idx = max_idx = None
                for row_idx, valid in enumerate(cur_exists):
                    if valid:
                        min_idx = row_idx - 1
                        break
                for row_idx, valid in enumerate(cur_exists[::-1]):
                    if valid:
                        max_idx = len(cur_exists) - row_idx
                        break
                if max_idx is not None:
                    max_idx = min(max_rows - 1, max_idx)
                if min_idx is not None:
                    min_idx = max(0, min_idx)
                lane_ranges.append([min_idx, max_idx])
            return lane_ranges

        batch_location_cls = preds_dicts['location_cls']      # (B, num_grids, N_row_anchors, N_lanes)
        batch_exist_cls = preds_dicts['exist_cls']    # (B, 2, N_row_anchors, N_lanes)
        batch_location_cls = batch_location_cls.detach().clone()
        batch_exist_cls = batch_exist_cls.detach().clone()

        results_list = []
        for location_cls, exist_cls, img_meta in zip(batch_location_cls, batch_exist_cls, img_metas):
            location_cls = location_cls.cpu().numpy()     # (num_grids, N_row_anchors, N_lanes)
            prob = scipy.special.softmax(location_cls, axis=0)    # (num_grids, N_row_anchors, N_lanes)
            grid_bins = np.arange(self.grid_num).reshape(-1, 1, 1)     # (num_grids, 1, 1)
            row_anchor_pred = np.sum(prob * grid_bins, axis=0)      # (N_row_anchors, N_lanes)

            # 获得预测车道的范围
            exist_cls = exist_cls.cpu().numpy()     # (2, N_row_anchors, N_lanes)
            exists = np.argmax(exist_cls, axis=0)      # (N_row_anchors, N_lanes)  # 1：说明存在， 0：不存在.
            lane_ranges = get_range(exists)      # List[[min_y, max_y], [min_y, max_y], ...]

            pred = self.predictions_to_pred(row_anchor_pred, lane_ranges, img_meta)
            results_list.append(pred)
        return results_list

    def predictions_to_pred(self, row_anchor_preds, lane_ranges, img_meta):
        """
        :param row_anchor_preds: (N_row_anchors, N_lanes)
        :param lane_ranges: List[[min_y, max_y], [min_y, max_y], ...]
        :param img_meta:
        :return:
            lanes: List[lane0, lane1, ...]
        """
        ori_height, ori_width = img_meta['ori_shape'][:2]
        img_h, img_w = img_meta['img_shape'][:2]
        crop = img_meta.get('crop', (0, 0, 0, 0))  # (x_min, y_min, x_max, y_max)
        y_min = crop[1]
        ratio_y = (ori_height - y_min) / img_h

        sample_y = self.test_cfg['sample_y']
        sample_y = np.array(sample_y, dtype=np.float32)
        lanes = []

        for lane_idx in range(row_anchor_preds.shape[1]):
            min_y, max_y = lane_ranges[lane_idx][0], lane_ranges[lane_idx][1]
            if min_y is None or max_y is None or (max_y - min_y) <= 2:
                continue

            # (N_valid_anchors,  )
            vaild_sample_y = sample_y[min_y:max_y]
            valid_row_anchor_pred = row_anchor_preds[min_y:max_y, lane_idx]

            # 缩放到原始图片的比例
            y = vaild_sample_y * ratio_y + y_min
            x = valid_row_anchor_pred * (ori_width - 1) / (self.grid_num - 1)

            coord = np.stack((x, y), axis=1)
            coord[:, 0] /= ori_width
            coord[:, 1] /= ori_height

            lanes.append(Lane(coord))

        return lanes