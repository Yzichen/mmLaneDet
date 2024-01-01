import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import cv2
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean
from mmlane.models import build_loss
from mmlane.core.evaluation import accuracy
from mmlane.models import HEADS
from mmlane.models.dense_heads import BaseDenseHead
from mmlane.core import build_assigner
from mmlane.ops import nms
from mmlane.core import Lane
from mmdet.core.utils import filter_scores_and_topk
from mmcv.cnn import (bias_init_with_prob, build_activation_layer,
                      build_norm_layer)
from mmcv.cnn import bias_init_with_prob, kaiming_init
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.builder import TRANSFORMER
import random
from mmcv.cnn import bias_init_with_prob, build_norm_layer, Linear
import copy
import time
from visualizer import get_local

from mmlane.models.aggregators import PyramidPoolingModule


class CtnetHead(nn.Module):
    def __init__(self, in_channels, heads_dict, final_kernel=1, init_bias=-2.19, use_bias=False):
        super(CtnetHead, self).__init__()
        self.heads_dict = heads_dict

        for cur_name in self.heads_dict:
            output_channels = self.heads_dict[cur_name]['out_channels']
            num_conv = self.heads_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU()
                ))
            fc_list.append(nn.Conv2d(in_channels, output_channels, kernel_size=final_kernel, stride=1,
                                     padding=final_kernel // 2, bias=True))
            fc = nn.Sequential(*fc_list)

            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m.weight.data)
                    if hasattr(m, "bias") and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.heads_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)
        return ret_dict


class ROIGather(nn.Module):
    '''
    ROIGather module for gather global information
    Args:
        in_channels: prior feature channels
        num_priors: prior numbers we predefined
        sample_points: the number of sampled points when we extract feature from line
        fc_hidden_dim: the fc output channel
        refine_layers: the total number of layers to build refine
    '''
    def __init__(self,
                 in_channels,
                 sample_points,
                 fc_hidden_dim,
                 ):
        super(ROIGather, self).__init__()
        self.in_channels = in_channels

        self.fc = nn.Linear(sample_points * in_channels, fc_hidden_dim)
        self.fc_norm = nn.LayerNorm(fc_hidden_dim)
        # self.attn = MultiheadAttention(embed_dims=self.in_channels,
        #                                num_heads=1,
        #                                attn_drop=0,
        #                                batch_first=True,
        #                                )
        # self.attention_norm = build_norm_layer(dict(type='LN'), fc_hidden_dim)[1]

    def forward(self, roi_features, x, n_pos):
        '''
        Args:
            roi_features: prior feature,  (B=1, n_pos, num_sample, C)
            x: feature map  (B, C, fH, fW)
            n_pos: int
        Return:
            roi: prior features with gathered global information, shape: (B*n_pos, C)
        '''
        roi = roi_features  # (B=1, n_pos, num_sample, C)
        bs = x.size(0)
        roi = roi.contiguous().view(bs * n_pos, -1)   # (B*num_priors, C*N_sample)
        # (B*num_priors, C*N_sample) --> (B*num_priors, C)
        roi = F.relu(self.fc_norm(self.fc(roi)))

        return roi


class SegDecoder(nn.Module):
    '''
    Optionaly seg decoder
    '''
    def __init__(self,
                 num_class,
                 prior_feat_channels=64,
                 num_layers=2):
        super().__init__()
        self.dropout = nn.Dropout2d(0.1)
        self.conv = nn.Conv2d(prior_feat_channels * num_layers, num_class, 1)

    def forward(self, x, img_h, img_w):
        """
        Args:
            x: (B, refine_layers*C, fH, fW)
        Returns:
            x: (B, num_class, img_H, img_W)
        """
        x = self.dropout(x)
        # (B, refine_layers*C, fH, fW) --> (B, num_class, fH, fW)
        x = self.conv(x)
        # (B, num_class, fH, fW) --> (B, num_class, img_H, img_W)
        x = F.interpolate(x,
                          size=[img_h, img_w],
                          mode='bilinear',
                          align_corners=False)
        return x


@HEADS.register_module()
class DALNetHead(BaseDenseHead):
    def __init__(self,
                 in_channel=64,
                 prior_feat_channels=64,
                 fc_hidden_dim=64,
                 num_points=72,
                 sample_points=36,
                 img_size=(800, 320),
                 num_stages=3,
                 num_classes=1,
                 seg_num_classes=6,
                 num_fc=2,
                 hm_scale=16,
                 radius=3,
                 hm_radius=2,
                 offset_radius=2,
                 theta_radius=2,
                 max_sample=5,
                 kernel_size=7,
                 proposal_head_dict={
                     'hm': {'out_channels': 1, 'num_conv': 2},
                     'offset': {'out_channels': 2, 'num_conv': 2},
                     'theta': {'out_channels': 1, 'num_conv': 2},
                 },
                 num_layers=2,
                 loss_heatmap=dict(
                     type='GaussianFocalLoss',
                     alpha=2.0,
                     gamma=4.0,
                     reduction='mean',
                     loss_weight=1.0
                 ),
                 loss_hm_reg=dict(
                     type='L1Loss',
                     reduction='mean',
                     loss_weight=2.0
                 ),
                 loss_hm_offset=dict(
                     type='SmoothL1Loss',
                     beta=1.0,
                     reduction='mean',
                     loss_weight=1.0
                 ),
                 loss_lane=dict(
                     type='SmoothL1Loss',
                     beta=1.0,
                     reduction='mean',
                     loss_weight=0.1
                 ),
                 loss_iou=dict(
                     type='LineIou_Loss',
                     length=15,
                     reduction='mean',
                     loss_weight=2.0
                 ),
                 loss_reg=dict(
                     type='SmoothL1Loss',
                     beta=1.0,
                     reduction='mean',
                     loss_weight=0.0
                 ),
                 loss_seg=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     ignore_index=255,
                     class_weight=1.0,
                     bg_cls_weight=0.4,
                     loss_weight=1.0
                 ),
                 sync_cls_avg_factor=True,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 share=True,
                 layer_weight=None,
                 with_seg=False
                 ):
        super(DALNetHead, self).__init__(init_cfg=init_cfg)
        self.in_channel = in_channel
        self.n_strips = num_points - 1
        self.n_offsets = num_points
        self.sample_points = sample_points
        self.img_w, self.img_h = img_size
        self.num_stages = num_stages
        self.num_classes = num_classes
        self.prior_feat_channels = prior_feat_channels
        self.fc_hidden_dim = fc_hidden_dim
        self.num_fc = num_fc
        self.hm_scale = hm_scale
        self.radius = radius
        self.hm_radius = hm_radius
        self.offset_radius = offset_radius
        self.theta_radius = theta_radius
        self.max_sample = max_sample
        self.kernel_size = kernel_size

        # 从num_points个点中采样sample_points个点，来pooling lane priors的特征.
        self.register_buffer(name='sample_x_indexs', tensor=(torch.linspace(
            0, 1, steps=self.sample_points, dtype=torch.float32) *
                                self.n_strips).long())
        self.register_buffer(name='prior_feat_ys', tensor=torch.flip(
            (1 - self.sample_x_indexs.float() / self.n_strips), dims=[-1]))
        self.register_buffer(name='prior_ys', tensor=torch.linspace(1, 0, steps=self.n_offsets, dtype=torch.float32))

        self.ctnet_head = CtnetHead(
            in_channels=self.in_channel,
            heads_dict=proposal_head_dict,
            final_kernel=1,
            init_bias=-2.19,
            use_bias=True)

        self.roi_gather = ROIGather(
            self.prior_feat_channels,
            self.sample_points,
            self.fc_hidden_dim,
        )

        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor

        self.loss_heatmap = build_loss(loss_heatmap)
        self.loss_hm_reg = build_loss(loss_hm_reg)
        self.loss_hm_offset = build_loss(loss_hm_offset)
        self.loss_lane = build_loss(loss_lane)
        self.loss_iou = build_loss(loss_iou)
        self.loss_reg = build_loss(loss_reg)

        self.test_cfg = test_cfg

        self.num_reg_fcs = 2
        self.share = share

        self._init_layers()
        self.init_weights()

        if layer_weight is not None:
            self.layer_weight = layer_weight
        else:
            self.layer_weight = [1.0] * self.num_stages
        self.layer_weight = nn.Parameter(torch.tensor(
            self.layer_weight, requires_grad=False), requires_grad=False)

        self.with_seg = with_seg
        if self.with_seg:
            self.use_sigmoid_seg = loss_seg.get('use_sigmoid', False)
            if self.use_sigmoid_seg:
                self.seg_out_channels = seg_num_classes
            else:
                self.seg_out_channels = seg_num_classes + 1

            # generate xys for feature map
            self.seg_decoder = SegDecoder(self.seg_out_channels,
                                          self.prior_feat_channels,
                                          num_layers)

            class_weight = loss_seg.get('class_weight', None)
            if class_weight is not None:
                class_weight = torch.ones(seg_num_classes + 1) * class_weight
                bg_cls_weight = loss_seg.get('bg_cls_weight', class_weight)
                assert isinstance(bg_cls_weight, float), 'Expected ' \
                                                         'bg_cls_weight to have type float. Found ' \
                                                         f'{type(bg_cls_weight)}.'
                class_weight[0] = bg_cls_weight     # 这里0是bg, 而mmdet系列是num_class是bg.
                loss_seg.update({'class_weight': class_weight})
                if 'bg_cls_weight' in loss_seg:
                    loss_seg.pop('bg_cls_weight')
                self.seg_bg_cls_weight = bg_cls_weight
            self.loss_seg = build_loss(loss_seg)
            self.seg_ignore_index = loss_seg.get('ignore_index', 255)

    def _init_layers(self):
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.fc_hidden_dim, self.fc_hidden_dim))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.fc_hidden_dim, self.n_offsets + 2))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
        if self.share:
            # 共享权重.
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(self.num_stages)])
        else:
            self.reg_branches = _get_clones(reg_branch, self.num_stages)

    # function to init layer weights
    def init_weights(self):
        for m in self.reg_branches:
            nn.init.constant_(m[-1].weight, 0)
            nn.init.constant_(m[-1].bias, 0)

    def pool_prior_features(self, batch_features, num_proposals, prior_xs):
        '''
        pool prior feature from feature map.
        Args:
            batch_features (Tensor): Input feature maps, shape: (B, C, fH, fW)
            num_proposals: int
            prior_xs: (B, num_proposals, N_sample)
        Returns:
            feature: (B, num_proposals, N_sample, C)
        '''
        batch_size = batch_features.shape[0]

        # (B, num_proposals, N_sample， 1)
        prior_xs = prior_xs.view(batch_size, num_proposals, -1, 1)
        # (N_sample, ) --> (B*N_sample*num_proposals, ) --> (B, num_proposals, N_sample, 1)
        prior_ys = self.prior_feat_ys.repeat(batch_size * num_proposals).view(
            batch_size, num_proposals, -1, 1)

        # (0, 1) --> (-1, 1)
        prior_xs = prior_xs * 2. - 1.
        prior_ys = prior_ys * 2. - 1.
        grid = torch.cat((prior_xs, prior_ys), dim=-1)      # (B, num_proposals, N_sample, 2)
        # (B, C, fH, fW) --> (B, C, num_proposals, N_sample) --> (B, num_proposals, N_sample, C)

        feature = F.grid_sample(batch_features, grid, align_corners=True).permute(0, 2, 3, 1).contiguous()

        return feature

    def generate_proposal(self, points, offsets, theta):
        """
        :param points: (N_pos, 2)  2:(x, y)
        :param offsets: (N_pos, 2)  2: (tx, ty)
        :param theta:  (N_pos, )
        :return:
            proposals: (N_pos, 2+S)
        """
        # 原图尺度
        points += offsets
        points *= self.hm_scale    # (N_pos, 2)

        points_y = 1 - (points[..., 1] / (self.img_h - 1))  # (N_pos, )
        points_x = points[..., 0] / (self.img_w - 1)  # (N_pos, )
        N_points, _ = points.shape

        # 1 start_y, 1 length, 72 coordinates,
        proposals = points.new_zeros((N_points, 2+self.n_offsets)).float()      # (N_pos, 2+S)
        proposals[..., 0] = points_y

        # start_x + ((1 - prior_y) - start_y) / tan(theta)     normalized x coords   按照图像底部-->顶部排列.
        proposals[..., 2:] = (points_x.unsqueeze(dim=-1).repeat(1, self.n_offsets) * (self.img_w - 1) +
                            ((1 - self.prior_ys.repeat(N_points, 1) -
                              points_y.unsqueeze(dim=-1).repeat(1, self.n_offsets)) * self.img_h /
                             torch.tan(theta.unsqueeze(dim=-1).repeat(1, self.n_offsets) * math.pi + 1e-5))) / (self.img_w - 1)

        # for vis
        # img = np.ones((self.img_h, self.img_w, 3), dtype=np.uint8) * 255
        # y_s = self.prior_ys * (self.img_h - 1)
        # vis_img = img.copy()
        # points = points.clone()
        # for idx, lane in enumerate(proposals):
        #     x_s = lane[2:] * (self.img_w - 1)
        #     color = (0, 0, 255)
        #     for i in range(1, self.n_offsets):
        #         cv2.line(vis_img, pt1=(x_s[i-1].int().item(), y_s[i-1].int().item()),
        #                  pt2=(x_s[i].int().item(), y_s[i].int().item()), color=color, thickness=2)
        #     cv2.circle(vis_img, (int(points[idx][0]), int(points[idx][1])), radius=4, color=(0, 255, 0),
        #                thickness=-1)
        #
        # cv2.imwrite("hm.png", vis_img)

        return proposals

    def forward_train(self,
                      x,
                      img_metas,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
        """
        batch_gt_lanes = kwargs['gt_lanes']     # List[(N_gt0, 4+S), (N_gt1, 4+S), ...]
        batch_gt_labels = kwargs['gt_labels']       # List[(N_gt0, ), (N_gt1, ), ...]
        batch_start_points = kwargs['start_points']     # List[(N_gt0, 2), (N_gt1, 2), ...]

        device = x[0].device
        poses, sampled_points, num_ins, gt_targets = self.parse_gt_infos(batch_gt_lanes,
                                                                   batch_gt_labels,
                                                                   batch_start_points,
                                                                   device=device)
        # sampled_poses: (N_pos,)
        # num_ins: [n_pos0, n_pos1, ...]
        # gt_targets: dict
        # {
        #     'sampled_gt_lanes': (N_pos, 4 + S)
        #     'sampled_gt_labels': (N_pos,)
        # }

        outs = self(x, poses, sampled_points, num_ins)
        losses = self.loss(outs, img_metas, gt_targets, **kwargs)
        return losses

    def parse_gt_infos(self, batch_gt_lanes, batch_gt_labels, batch_start_points, device):
        """
        :param:
            batch_gt_lanes: List[(N_gt0, 4+S), (N_gt1, 4+S), ...]
            batch_gt_labels: List[(N_gt0, ), (N_gt1, ), ...]
            batch_start_points: List[(N_gt0, 2), (N_gt1, 2), ...]
            hm_scale: int
        :return:
            sampled_poses: (N_pos, )
            sampled_start_ps: (N_pos, 2)
            num_ins: [n_pos0, n_pos1, ...]
            gt_targets: dict{
                'sampled_gt_lanes': (N_pos, 4+S)
                'sampled_gt_labels': (N_pos, )
            }

        """
        def cal_dis(p1, p2):
            return torch.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        def select_mask_points(ct, r, shape, max_sample=5):
            """
            从lane起始点周围选取max_sample个点, 后续应当是取这些点对应的 kernel params 进行lane shape预测, 提供正确的训练样本.
            Args:
                ct: (center_x, center_y)
                r: float
                shape: (hm_h, hm_w)
            Returns:
                sample_points: [(x0, y0), (x1, y1), ..]
            """

            def in_range(pt, w, h):
                if pt[0] >= 0 and pt[0] < w and pt[1] >= 0 and pt[1] < h:
                    return True
                else:
                    return False

            h, w = shape[:2]
            ct[0] = min(ct[0], w-1)
            ct[1] = min(ct[1], h-1)
            sample_points = []
            start_x, end_x = ct[0] - r, ct[0] + r
            start_y, end_y = ct[1] - r, ct[1] + r
            for x in range(start_x, end_x + 1):
                for y in range(start_y, end_y + 1):
                    if x == ct[0] and y == ct[1]:
                        continue
                    if in_range((x, y), w, h) and cal_dis((x, y), ct) <= r + 0.1:
                        sample_points.append([x, y])
            if len(sample_points) > max_sample - 1:
                sample_points = random.sample(sample_points, max_sample - 1)
            sample_points.append([int(ct[0]), int(ct[1])])
            return sample_points

        n_cls = self.num_classes
        hm_h, hm_w = self.img_h / self.hm_scale, self.img_w / self.hm_scale

        sampled_poses = []
        sampled_points = []
        sampled_gt_lanes = []
        sampled_gt_lables = []
        num_ins = []
        for batch_idx in range(len(batch_gt_lanes)):
            cur_gt_lanes = batch_gt_lanes[batch_idx]    # (N_gt, 4+S)
            cur_gt_labels = batch_gt_labels[batch_idx]      # (N_gt, )
            cur_start_points = batch_start_points[batch_idx]    # (N_gt, 2)    2: (x, y)

            gt_hm_lane_start = cur_start_points / self.hm_scale
            gt_hm_lane_start_int = gt_hm_lane_start.round().int()    # (N_gt, 2)    2: (x, y)

            num = 0
            for lane_idx in range(len(cur_gt_lanes)):
                cur_gt_lane = cur_gt_lanes[lane_idx]    # (4+S, )
                cur_gt_label = cur_gt_labels[lane_idx]      # (1, )
                cur_sample_points = select_mask_points(ct=gt_hm_lane_start_int[lane_idx],
                                                       r=self.radius,
                                                       shape=(hm_h, hm_w),
                                                       max_sample=self.max_sample
                                                       )    # [(x0, y0), (x1, y1), ..]   len = N_sample

                new_sample_points = []
                for sample_point in copy.deepcopy(cur_sample_points):
                    x, y = sample_point[0], sample_point[1]
                    for gt_id in range(len(cur_start_points)):
                        if gt_id == lane_idx:
                            continue
                        s_points = gt_hm_lane_start[gt_id]
                        if cal_dis((x, y), s_points) <= self.radius + 0.1:
                            continue
                        else:
                            new_sample_points.append(sample_point)
                cur_sample_points = new_sample_points

                num += len(cur_sample_points)
                # 从heat_map(B, C=1, hm_H, hm_W)中取对应点位置[x, y]的theta, 然后进行lane shape预测, 提供正确的训练样本.
                for p in cur_sample_points:
                    pos = batch_idx * n_cls * hm_h * hm_w + cur_gt_label * hm_h * hm_w + p[1] * hm_w + p[0]
                    sampled_poses.append(pos.item())
                    sampled_gt_lanes.append(cur_gt_lane)
                    sampled_gt_lables.append(cur_gt_label)
                    sampled_points.append(p)

            num_ins.append(num)

        if len(sampled_poses) > 0:
            sampled_poses = torch.from_numpy(np.array(sampled_poses, dtype=np.long)).to(device)
            sampled_gt_lanes = torch.stack(sampled_gt_lanes, dim=0)         # (N_pos, 4+S)
            sampled_gt_labels = torch.stack(sampled_gt_lables, dim=0)       # (N_pos, )
            sampled_points = torch.from_numpy(np.array(sampled_points, dtype=np.float32)).to(device)     # (N_pos, 2)
        else:
            sampled_poses = torch.zeros((0, ), dtype=torch.long).to(device)
            sampled_gt_lanes = torch.zeros((0, 4+self.n_offsets), dtype=torch.float32).to(device)
            sampled_gt_labels = torch.zeros((0, ), dtype=torch.long).to(device)
            sampled_points = torch.zeros((0, 2), dtype=torch.float32).to(device)

        gt_targets = {
            'sampled_gt_lanes': sampled_gt_lanes,       # (N_pos, 4+S)
            'sampled_gt_labels': sampled_gt_labels      # (N_pos, )
        }

        return sampled_poses, sampled_points, num_ins, gt_targets

    def parse_pos(self, seeds, hm_shape):
        """
        Args:
            seeds: List[dict{
                        'coord': coord,  # [x, y]
                        'id_class': id_class,   # cls
                        'score': score
                    }, ...]     # len = N_pos
            hm_shape: (h, w)
        Returns:
            poses:  List[p0, p1, ...]
            start_points: List[(sx0, sy0), (sx1, sy1), ...]
        """
        hm_h, hm_w = hm_shape
        pose = []
        start_points = []
        for seed in seeds:
            x, y = seed['coord']
            label = seed['id_class']
            pos = label * hm_h * hm_w + y * hm_w + x
            pose.append(pos)
            start_points.append((x, y))

        return pose, start_points

    @get_local('heat')
    def ctdet_decode(self, heat, thr=0.1):
        """
        :param hm: (1, h_hm, w_hm)
        :param thr: float
        :param max_lanes: int
        :return:  seeds: List[dict{
                    'coord': coord,  # [x, y]
                    'id_class': id_class,   # cls+1
                    'score': score
                 }, ...]     # len = N_pos
        """
        def _nms(heat, kernel=5):
            if isinstance(kernel, int):
                pad = (kernel - 1) // 2
                hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
            else:
                pad = ((kernel[0] - 1) // 2, (kernel[1] - 1) // 2)
                hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
            keep = (hmax == heat).float()
            return heat * keep

        def _format(heat, inds):
            """
            Args:
                heat: (h_hm, w_hm, n_cls(1))
                inds: Tuple(h_id(n_pos, ), w_id(n_pos, ),  cls_id(n_pos, ))
            """
            ret = []
            for y, x, c in zip(inds[0], inds[1], inds[2]):
                id_class = c
                coord = [x, y]
                score = heat[y, x, c]
                ret.append({
                    'coord': coord,  # [x, y]
                    'id_class': id_class,   # cls
                    'score': score
                })
            return ret

        # (n_cls(1), h_hm, w_hm)
        heat_nms = _nms(heat, kernel=self.kernel_size)
        # (n_cls(1), h_hm, w_hm) --> (h_hm, w_hm, n_cls(1))
        heat_nms = heat_nms.permute(1, 2, 0).contiguous()
        # Tuple(h_id(n_pos, ), w_id(n_pos, ),  cls_id(n_pos, ))
        inds = torch.where(heat_nms > thr)
        seeds = _format(heat_nms, inds)
        return seeds

    # forward function here
    def forward(self, x, sampled_poses=None, sampled_points=None, num_ins=None):
        '''
        Args:
            x: input features (list[Tensor])  List[(B, C, H2, W2), (B, C, H4, W4)]
            sampled_poses: (N_pos, )
            sampled_points: (N_pos, 2)
            num_ins: [n_pos0, n_pos1, ...]
        Return:
            output: dict{
                'all_cls_scores': (num_layers, B, num_proposals, n_cls)
                'all_lanes_preds': (num_layers, B, num_proposals, 2+S)
                'hm': (B, 1, h_hm, w_hm),
                'offset': (B, 2, h_hm, w_hm),
                'theta': (B, 1, h_hm, w_hm)
            }
        '''
        if len(x) == 2:
            f_position, f_hm = x[0], x[1]   # (B, C, H2, W2),  (B, C, H4, W4)
            # f_position, f_hm = x[1], x[0]  # (B, C, H2, W2),  (B, C, H4, W4)
        else:
            f_position, f_hm = x[0], x[0]   # (B, C, H2, W2),  (B, C, H4, W4)
        batch_size, C, h_hm, w_hm = f_hm.shape

        # f_hm = self.ppm(f_hm)
        result = self.ctnet_head(f_hm)
        # result: dict{
        #     'hm': (B, 1, h_hm, w_hm),
        #     'offset': (B, 2, h_hm, w_hm),
        #     'theta': (B, 1, h_hm, w_hm)
        # }
        if self.training:
            with torch.no_grad():
                # (B, 1, h_hm, w_hm) --> (B, h_hm, w_hm, 1) --> (B*h_hm*w_hm, 1)
                theta = result['theta'].permute(0, 2, 3, 1).contiguous().view(batch_size*h_hm*w_hm, -1)
                # (B, 2, h_hm, w_hm) --> (B, h_hm, w_hm, 2) --> (B*h_hm*w_hm, 2)
                offset = result['offset'].permute(0, 2, 3, 1).contiguous().view(batch_size*h_hm*w_hm, -1)
                theta = theta.squeeze()     # (B*h_hm*w_hm, 1) --> (B*h_hm*w_hm, )

                # (B*h_hm*w_hm, ) --> (N_pos, )
                sampled_theta = torch.gather(theta, dim=0, index=sampled_poses)

                # (B*h_hm*w_hm, 2) --> (N_pos, 2)
                sampled_offsets = torch.gather(offset, dim=0, index=sampled_poses.unsqueeze(dim=-1).repeat(1, 2))

                # (B, C, h_hm, w_hm) --> (B, h_hm, w_hm, C) --> (B*h_hm*w_hm, C)
                proposals_lane = self.generate_proposal(sampled_points, sampled_offsets, sampled_theta)    # (N_pos, 2+S)

            priors_on_featmap = proposals_lane.clone()[..., 2 + self.sample_x_indexs]   # (N_pos, num_sample)

            # List[(n_pos0, 2+S), (n_pos1, 2+S), ...]
            proposals_lane = torch.split(proposals_lane, num_ins, dim=0)
            # List[(n_pos0, num_sample), (n_pos1, num_sample), ...]
            priors_on_featmap = torch.split(priors_on_featmap, num_ins, dim=0)

            batch_lanes = []
            for batch_id in range(batch_size):
                out_lanes = []

                cur_num_proposals = num_ins[batch_id]   # n_pos
                if cur_num_proposals == 0:
                    batch_lanes.append(out_lanes)
                    continue
                cur_f_position = f_position[batch_id:batch_id + 1, ...]  # (1, C, H2, W2)
                cur_proposals_lane = proposals_lane[batch_id]   # (n_pos, 2+S)
                cur_priors_on_featmap = priors_on_featmap[batch_id]     # (n_pos, num_sample)

                for stage in range(self.num_stages):
                    cur_prior_xs = torch.flip(cur_priors_on_featmap, dims=[1])  # 图像顶部-->底部  (N_pos, num_sample)
                    roi_features = self.pool_prior_features(
                        cur_f_position, cur_num_proposals, cur_prior_xs)    # (B=1, n_pos, num_sample, C)

                    fc_features = self.roi_gather(roi_features, cur_f_position, cur_num_proposals)      # (B, n_pos, C)
                    reg_features = fc_features.clone()

                    cur_lane_pred = self.reg_branches[stage](reg_features)

                    cur_predictions = cur_proposals_lane.clone()    # (n_pos, 2+S)
                    cur_predictions[:, 0] += cur_lane_pred[:, 0]    # 1 start_y,
                    cur_predictions[:, 1] = cur_lane_pred[:, 1]     # length, (n_pos, )
                    cur_predictions[:, 2:] += cur_lane_pred[:, 2:]

                    out_lanes.append(cur_predictions)   # List[(n_pos0, 2+S), (n_pos0, 2+S), ...]

                    if stage != self.num_stages - 1:
                        # cur_proposals_lane = cur_predictions.detach().clone()
                        cur_proposals_lane = cur_predictions
                        cur_priors_on_featmap = cur_proposals_lane[:, 2+self.sample_x_indexs]

                batch_lanes.append(out_lanes)

            all_lanes_preds = []
            for layer_id in range(self.num_stages):
                cur_lanes_preds = []
                for batch_id in range(batch_size):
                    if len(batch_lanes[batch_id]) == 0:
                        continue
                    cur_lanes_preds.append(batch_lanes[batch_id][layer_id])    # (n_pos, 2+S)
                cur_lanes_preds = torch.cat(cur_lanes_preds, dim=0)     # (N_pos, 2+S)
                all_lanes_preds.append(cur_lanes_preds)

            all_lanes_preds = torch.stack(all_lanes_preds, dim=0)   # (num_layers, N_pos, 2+S)

            output = {
                'all_lanes_preds': all_lanes_preds,  # (num_layers, N_pos, 2+S)
            }
            output.update(result)

            if self.with_seg:
                # (B, 2*C, H3, W3)
                seg_features = torch.cat([
                    F.interpolate(feature,
                                  size=[
                                      x[-1].shape[2],
                                      x[-1].shape[3]
                                  ],
                                  mode='bilinear',
                                  align_corners=False)
                    for feature in x], dim=1)
                # (B, 2s*C, H3, W3)  --> (B, num_class, img_H, img_W)
                seg = self.seg_decoder(seg_features, self.img_h, self.img_w)
                output['pred_seg'] = seg

            return output
        else:
            hms = result['hm']      # (B, 1, hm_h, hm_w)
            thetas = result['theta']    # (B, 1, hm_h, hm_w)
            offsets = result['offset']  # (B, 2, hm_h, hm_w)
            hms = torch.clamp(hms.sigmoid(), min=1e-4, max=1 - 1e-4)  # (B, 1, H4, W4)

            batch_size = hms.shape[0]
            out_seeds = []
            for batch_id in range(batch_size):
                cur_hm = hms[batch_id]  # (1, hm_h, hm_w)
                cur_thetas = thetas[batch_id]   # (1, hm_h, hm_w)
                cur_offsets = offsets[batch_id]  # (2, hm_h, hm_w)
                cur_f_position = f_position[batch_id:batch_id+1, ...]   # (B=1, C, H2, W2)
                seeds = self.ctdet_decode(cur_hm, thr=self.test_cfg['hm_thr'])
                # seeds: List[dict{
                #     'coord': coord,  # [x, y]
                #     'id_class': id_class,  # cls
                #     'score': score
                # }, ...],    len = N_pos
                cur_num_proposals = len(seeds)

                if len(seeds):
                    # poses: List[p0, p1, ...]
                    # start_points: List[(sx0, sy0), (sx1, sy1), ...]
                    pose, start_points = self.parse_pos(seeds, (h_hm, w_hm))  # (N_pos, )
                    pose = torch.tensor(pose, dtype=torch.long, device=hms.device)  # (N_pos, )
                    pos_start_points = torch.tensor(start_points, dtype=torch.float32, device=hms.device)   # (N_pos, 2)

                    # (1, hm_h, hm_w) --> (hm_h, hm_w, 1) --> (hm_h*hm_w, )
                    cur_thetas = cur_thetas.permute(1, 2, 0).contiguous().view(h_hm*w_hm)
                    # (h_hm*w_hm, ) --> (N_pos, )
                    pos_theta = torch.gather(cur_thetas, dim=0, index=pose)

                    # (2, hm_h, hm_w) --> (hm_h, hm_w, 2) --> (hm_h*hm_w, 2)
                    cur_offsets = cur_offsets.permute(1, 2, 0).contiguous().view(h_hm * w_hm, -1)
                    # (h_hm*w_hm, 2) --> (N_pos, 2)
                    pos_offset = torch.gather(cur_offsets, dim=0, index=pose.unsqueeze(dim=-1).repeat(1, 2))

                    cur_proposals_lane = self.generate_proposal(pos_start_points, pos_offset, pos_theta)  # (N_pos, 2+S)
                    cur_priors_on_featmap = cur_proposals_lane.clone()[..., 2 + self.sample_x_indexs]   # (N_pos, num_sample)

                    out_lanes = []
                    for stage in range(self.num_stages):
                        cur_prior_xs = torch.flip(cur_priors_on_featmap, dims=[1])  # 图像顶部-->底部  (N_pos, num_sample)

                        roi_features = self.pool_prior_features(
                            cur_f_position, cur_num_proposals, cur_prior_xs)    # (B=1, n_pos, num_sample, C)

                        fc_features = self.roi_gather(roi_features, cur_f_position, cur_num_proposals)  # (B, n_pos, C)
                        reg_features = fc_features.clone()

                        cur_lane_pred = self.reg_branches[stage](reg_features)

                        cur_predictions = cur_proposals_lane.clone()  # (n_pos, 2+S)
                        cur_predictions[:, 0] += cur_lane_pred[:, 0]  # 1 start_y
                        cur_predictions[:, 1] = cur_lane_pred[:, 1]
                        cur_predictions[:, 2:] += cur_lane_pred[:, 2:]
                        out_lanes.append(cur_predictions)  # List[(n_pos0, 2+S), (n_pos0, 2+S), ...]

                        if stage != self.num_stages - 1:
                            cur_proposals_lane = cur_predictions.detach().clone()
                            cur_priors_on_featmap = cur_proposals_lane[:, 2 + self.sample_x_indexs]

                    for seed_id in range(len(seeds)):
                        # List[(2+S, ), (2+S, ), ...]
                        seeds[seed_id]['pred_lane'] = [lane[seed_id] for lane in out_lanes]
                        seeds[seed_id]['anchor'] = cur_proposals_lane[seed_id]
                        seeds[seed_id]['start_points'] = pos_start_points[seed_id]
                        seeds[seed_id]['theta'] = pos_theta[seed_id]
                else:
                    seeds = []

                out_seeds.append(seeds)

            output = {
                'seeds': out_seeds
            }
            return output

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             preds_dicts,
             img_metas,
             gt_targets,
             gt_lanes,
             gt_labels,
             start_points,
             gt_semantic_seg=None,
             **kwargs
             ):
        """Compute losses of the head.
        Args:
            preds_dicts: dict{
                'all_lanes_preds':  (num_layers, N_pos, 2+S)
                    2+S: 1 start_y(normalized), 1 length(normalized), S coordinates(normalized),
                'hm': (B, 1, h_hm, w_hm),
                'offset': (B, 2, h_hm, w_hm),
                'theta': (B, 1, h_hm, w_hm)
            }
            gt_lanes (List):  List[(N_lanes0, 4+S), (N_lanes1, 4+S), ...]
                4+S: 1 start_y (normalized), 1 start_x (absolute), 1 theta, 1 length (absolute),
                     S coordinates(absolute)
            gt_labels (List): List[(N_lanes0, ), (N_lanes1, ), ...]
            start_points: List[(N_lanes0, 2), (N_lanes1, 2), ...]
            gt_targets: dict
            {
                'sampled_gt_lanes': (N_pos, 2 + S)
                'sampled_gt_labels': (N_pos,)
            }

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        loss_dict = dict()
        gt_thetas = [gt_lane[:, 2] for gt_lane in gt_lanes]  # List[(N_lanes0, ), (N_lanes1, ), ...]
        hm_target = self.get_hm_targets(start_points, gt_thetas)
        gt_heatmap, gt_offset, gt_theta, theta_masks, offset_masks = hm_target
        # gt_heatmap: (B, h_hm, w_hm)
        # gt_offset: (B, h_hm, w_hm, 2)
        # gt_theta:  (B, h_hm, w_hm)
        # theta_masks: (B, h_hm, w_hm)
        # offset_masks: (B, h_hm, w_hm, 2)

        pred_heatmap = preds_dicts['hm']  # (B, 1, h_hm, w_hm)
        pred_offset = preds_dicts['offset']     # (B, 2, h_hm, w_hm)
        pred_theta = preds_dicts['theta']  # (B, 1, h_hm, w_hm)

        pred_heatmap = torch.clamp(pred_heatmap.sigmoid(), min=1e-4, max=1 - 1e-4)  # (B, 1, H4, W4)
        gt_heatmap = gt_heatmap.unsqueeze(dim=1)  # (B, 1, h_hm, w_hm)
        avg_factor = max(1, gt_heatmap.eq(1).sum())
        loss_heatmap = self.loss_heatmap(pred_heatmap, gt_heatmap, avg_factor=avg_factor)

        pred_offset = pred_offset.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        gt_offset = gt_offset.view(-1, 2)
        offset_masks = offset_masks.view(-1, 2)
        loss_offset = self.loss_hm_offset(
            pred_offset, gt_offset, weight=offset_masks, avg_factor=offset_masks.sum())

        pred_theta = pred_theta.permute(0, 2, 3, 1).contiguous().view(-1)
        gt_theta = gt_theta.view(-1)
        theta_masks = theta_masks.view(-1)
        loss_theta = self.loss_hm_reg(
            pred_theta, gt_theta, weight=theta_masks, avg_factor=theta_masks.sum())

        loss_dict['loss_heatmap'] = loss_heatmap
        loss_dict['loss_offset'] = loss_offset
        loss_dict['loss_theta'] = loss_theta

        all_lanes_preds = preds_dicts['all_lanes_preds']  # (num_layers, N_pos, 2+S)

        # List[(N_pos, 4+S), (N_pos, 4+S), ...]
        all_gt_lanes_list = [gt_targets['sampled_gt_lanes'] for _ in range(self.num_stages)]

        # 分别计算每层layer的loss
        losses_yl, losses_iou, losses_reg = multi_apply(self.loss_single, all_lanes_preds, all_gt_lanes_list)

        # loss from the last decoder layer
        loss_dict['loss_yl'] = losses_yl[-1] * self.layer_weight[-1]
        loss_dict['loss_iou'] = losses_iou[-1] * self.layer_weight[-1]
        loss_dict['loss_reg'] = losses_reg[-1] * self.layer_weight[-1]

        # loss from other decoder layers
        num_dec_layers = 0
        for loss_yl_i, loss_iou_i, loss_reg_i, layer_weight_i in zip(losses_yl[:-1], losses_iou[:-1], losses_reg[:-1],
                                                                     self.layer_weight[:-1]):
            loss_dict[f'd{num_dec_layers}.loss_yl'] = loss_yl_i * layer_weight_i
            loss_dict[f'd{num_dec_layers}.loss_iou'] = loss_iou_i * layer_weight_i
            loss_dict[f'd{num_dec_layers}.loss_reg'] = loss_reg_i * layer_weight_i
            num_dec_layers += 1

        if self.with_seg:
            pred_seg = preds_dicts['pred_seg']      # (B, num_class, img_H, img_W)
            gt_semantic_seg = gt_semantic_seg.squeeze(dim=1).long()      # (B, img_H, img_W)
            num_total_pos = (torch.logical_and(gt_semantic_seg > 0, gt_semantic_seg != self.seg_ignore_index)).sum()
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

    def loss_single(self, lanes_preds, gt_lanes):
        """
        :param
            lanes_preds:  (N_pos, 2+S)
                4+S: 1 start_y(normalized), 1 length(normalized),
                     S coordinates(normalized),
            gt_lanes: (N_pos, 4+S)
                4+S: 1 start_y(normalized), ..., 1 length(absolute),
                     S coordinates(absolute)
            gt_labels: (N_pos, )
        :return:
        """
        num_total_pos = lanes_preds.shape[0]

        # regression loss
        lanes_preds = lanes_preds.view(-1, lanes_preds.size(-1))  # (N_pos, 2+S)
        lane_weights = torch.ones_like(lanes_preds)  # (N_pos, 2+S)
        # yxtl regression loss
        # (N_pos, 2)
        # start_y(normalized), 1 length(normalized)
        # normalized --> absolute
        pred_yl = lanes_preds[..., :2].clone()
        pred_yl[:, 0] *= self.n_strips
        pred_yl[:, 1] *= self.n_strips

        # (N_pos, 2)
        # start_y(normalized), 1 length(absolute)
        target_yl = torch.stack((gt_lanes[:, 0], gt_lanes[:, 3]), dim=-1)
        # normalized --> absolute
        target_yl[:, 0] *= self.n_strips

        # 调整length
        with torch.no_grad():
            pred_start_y = torch.clamp(pred_yl[:, 0].round().long(), 0, self.n_strips)
            target_start_y = target_yl[:, 0].round().long()
            target_yl[:, -1] -= (pred_start_y - target_start_y)

        loss_yl = self.loss_lane(
            pred_yl,
            target_yl,
            weight=lane_weights[..., :2],
            avg_factor=num_total_pos * 2,
        )

        # line Iou loss (based on x_coords)
        # (N_pos, S)
        # normalized --> absolute
        pred_x_coords = lanes_preds[..., 2:].clone() * (self.img_w - 1)
        # (N_pos, S)  absolute
        target_x_coords = gt_lanes[..., 4:]

        loss_iou = self.loss_iou(
            pred_x_coords,
            target_x_coords,
            self.img_w,
            weight=lane_weights[..., 0],
            avg_factor=num_total_pos,
        )

        valid_mask = (target_x_coords >= 0) & (target_x_coords < self.img_w)   # (B*num_priors, S)
        weight = lane_weights[..., -self.n_offsets:] * valid_mask
        loss_reg = self.loss_reg(
            pred_x_coords,
            target_x_coords,
            weight,
            avg_factor=weight.sum()
        )

        return loss_yl, loss_iou, loss_reg

    def get_hm_targets(self, start_points, gt_thetas):
        """
        :param start_points: List[(N_lanes0, 2), (N_lanes1, 2), ...]
        :param gt_thetas: List[(N_lanes0, ), (N_lanes1, ), ...]
        :return:
        """
        def draw_umich_gaussian(heatmap, center, radius, k=1):
            """
            Args:
                heatmap: (hm_h, hm_w)   1/16
                center: (x0', y0'),  1/16
                radius: float
            Returns:
                heatmap: (hm_h, hm_w)
            """
            def gaussian2D(shape, sigma=1):
                """
                Args:
                    shape: (diameter=2*r+1, diameter=2*r+1)
                Returns:
                    h: (diameter, diameter)
                """
                m, n = [(ss - 1.) / 2. for ss in shape]
                # y: (1, diameter)    x: (diameter, 1)
                y, x = np.ogrid[-m:m + 1, -n:n + 1]
                # (diameter, diameter)
                h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
                h[h < np.finfo(h.dtype).eps * h.max()] = 0
                return h

            diameter = 2 * radius + 1
            # (diameter, diameter)
            gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
            x, y = center[0].round(), center[1].round()
            height, width = heatmap.shape[0:2]
            x = int(min(x, width-1))
            y = int(min(y, height-1))
            left, right = min(x, radius), min(width - x, radius + 1)
            top, bottom = min(y, radius), min(height - y, radius + 1)
            masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
            masked_gaussian = torch.from_numpy(
                gaussian[radius - top:radius + bottom, radius - left:radius + right]
            ).to(heatmap.device).float()
            if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
                torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
            return heatmap

        def draw_theta_map(theta_map, gt_theta, mask, pos_float, radius):
            pos_x, pos_y = pos_float[0].round(), pos_float[1].round()
            height, width = theta_map.shape[0:2]
            x = int(min(pos_x, width-1))
            y = int(min(pos_y, height-1))
            left, right = min(x, radius), min(width - x, radius + 1)
            top, bottom = min(y, radius), min(height - y, radius + 1)

            for i in range(y-top, y+bottom):
                for j in range(x-left, x+right):
                    # if torch.sqrt((pos_float[0] - j)**2 + (pos_float[1] - i)**2) < radius:
                    if mask[i, j] == 0:
                        theta_map[i, j] = gt_theta
                        mask[i, j] = 1
                    else:
                        mask[i, j] = 0

        def draw_offset_map(offset_map, mask, pos_float, radius):
            pos_x, pos_y = pos_float[0].round(), pos_float[1].round()
            height, width = offset_map.shape[0:2]
            x = int(min(pos_x, width-1))
            y = int(min(pos_y, height-1))
            left, right = min(x, radius), min(width - x, radius + 1)
            top, bottom = min(y, radius), min(height - y, radius + 1)

            for i in range(y-top, y+bottom):
                for j in range(x-left, x+right):
                    # if torch.sqrt((pos_float[0] - j)**2 + (pos_float[1] - i)**2) < radius:
                    if mask[i, j, 0] == 0:
                        offset_map[i, j, 0] = pos_float[0] - j
                        offset_map[i, j, 1] = pos_float[1] - i
                        mask[i, j] = 1
                    else:
                        mask[i, j] = 0

        assert len(start_points) == len(gt_thetas)
        hm_scale = self.hm_scale
        h_hm, w_hm = self.img_h // hm_scale, self.img_w // hm_scale
        batch_size = len(start_points)
        hm_radius = self.hm_radius
        offset_radius = self.offset_radius
        theta_radius = self.theta_radius

        heatmap_list = []
        gt_offset_list = []
        gt_theta_list = []
        theta_mask_list = []
        offset_mask_list = []
        for batch_id in range(batch_size):
            gt_heatmap = torch.zeros((int(h_hm), int(w_hm)), dtype=torch.float, device=start_points[0].device)
            gt_offset = torch.zeros((int(h_hm), int(w_hm), 2), dtype=torch.float, device=start_points[0].device)
            gt_theta = torch.zeros((int(h_hm), int(w_hm)), dtype=torch.float, device=start_points[0].device)
            theta_mask = torch.zeros((int(h_hm), int(w_hm)), dtype=torch.long, device=start_points[0].device)
            offset_mask = torch.zeros((int(h_hm), int(w_hm), 2), dtype=torch.long, device=start_points[0].device)

            cur_start_points = start_points[batch_id]   # (N, 2)  2:(x, y)
            cur_gt_thetas = gt_thetas[batch_id]     # (N, )
            for k in range(len(cur_start_points)):   # point: (x, y)
                point = cur_start_points[k]     # (x, y)
                gt_hm_lane_start = (point[0] / hm_scale, point[1] / hm_scale)
                draw_umich_gaussian(gt_heatmap, gt_hm_lane_start, radius=hm_radius)
                theta = cur_gt_thetas[k]    # float

                draw_theta_map(gt_theta, theta, theta_mask, gt_hm_lane_start, radius=theta_radius)
                draw_offset_map(gt_offset, offset_mask, gt_hm_lane_start, radius=offset_radius)

            # heat_map = gt_heatmap.detach().cpu().numpy() * 255
            # heat_map = heat_map.astype(np.uint8)
            # # cv2.imwrite("gt_hm.png", heat_map)
            # theta = gt_theta.detach().cpu().numpy()

            # offset = gt_offset.detach().cpu().numpy()
            # offset_x = offset[..., 0]
            # offset_y = offset[..., 1]

            heatmap_list.append(gt_heatmap)
            gt_offset_list.append(gt_offset)
            gt_theta_list.append(gt_theta)
            theta_mask_list.append(theta_mask)
            offset_mask_list.append(offset_mask)

        gt_heatmap = torch.stack(heatmap_list, dim=0)   # (B, h_hm, w_hm)
        gt_offset = torch.stack(gt_offset_list, dim=0)  # (B, h_hm, w_hm, 2)
        gt_theta = torch.stack(gt_theta_list, dim=0)    # (B, h_hm, w_hm)
        theta_masks = torch.stack(theta_mask_list, dim=0)     # (B, h_hm, w_hm)
        offset_masks = torch.stack(offset_mask_list, dim=0)  # (B, h_hm, w_hm)

        return gt_heatmap, gt_offset, gt_theta, theta_masks, offset_masks

    def nms_seeds_tiny(self, seeds, thr):
        """
        Args:
            seeds: List[dict{
                'coord': [x, y]
                'id_class': id_class,  # cls+1
                'score': float,
                'mask': (H2, W2),
                'offset': (H2, W2),
                'range': (2, H2)
            }, ...]
        Returns:
            update_seeds: List[dict{
                'coord': [x, y]
                'id_class': id_class,  # cls+1
                'score': float,
                'mask': (H2, W2),
                'offset': (H2, W2),
                'range': (2, H2)
            }, ...]
        """

        def cal_dis(p1, p2):
            return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        def search_groups(coord, groups, thr):
            for idx_group, group in enumerate(groups):
                for group_point in group:
                    group_point_coord = group_point[1]
                    if cal_dis(coord, group_point_coord) <= thr:
                        return idx_group
            return -1

        def choose_highest_score(group):
            highest_score = -1
            highest_idx = -1
            for idx, _, score in group:
                if score > highest_score:
                    highest_idx = idx
            return highest_idx

        def update_coords(points_info, thr=4):
            """
            Args:
                points_info: List[([x, y], float), ([x, y], float), ...]
                thr: int
            Returns:
                keep_idx: List[idx0, idx1, ...]
            """
            groups = []
            keep_idx = []
            # 将所有seed点根据距离阈值thr 划分到不同的group中.
            for idx, (coord, score) in enumerate(points_info):
                # idx：int, coord: [x, y];  score: float
                idx_group = search_groups(coord, groups, thr)
                if idx_group < 0:
                    groups.append([(idx, coord, score)])
                else:
                    groups[idx_group].append((idx, coord, score))

            # 在每个group中选择score最大的seed点.
            for group in groups:
                # group: List[(idx, coord, score), ...]
                choose_idx = choose_highest_score(group)
                if choose_idx >= 0:
                    keep_idx.append(choose_idx)
            return keep_idx

        # List[([x, y], float), ([x, y], float), ...]
        points = [(item['coord'], item['score']) for item in seeds]
        # 根据score和距离阈值thr， 过滤掉一部分seed点.
        keep_idxes = update_coords(points, thr=thr)
        update_seeds = [seeds[idx] for idx in keep_idxes]
        return update_seeds

    def collect_seeds(self, seeds):
        """
        Args:
            seeds: List[dict
            {
                'coord': [x, y]
                'id_class': id_class,  # cls+1
                'score': float,
                'pred_lane': List[(4+S, ), (4+S, ), (4+S, )]
            }, ...]
        Returns:
            scores: List[float, float, ...]
            ranges: (N_pos, 2, H2)
        """
        scores = []
        labels = []
        pred_lanes = []
        for seed in seeds:
            scores.append(seed['score'])
            labels.append(seed['id_class'])
            pred_lanes.append(seed['pred_lane'][-1])

        if len(scores) > 0:
            scores = torch.stack(scores, 0)  # (N_pos, )
            labels = torch.stack(labels, 0)  # (N_pos, )
            pred_lanes = torch.stack(pred_lanes, 0)  # (N_pos, 4+S)
            return scores, labels, pred_lanes
        else:
            return None

    def get_lanes(self, output, img_metas):
        """
        Args:
            output: dict
            {
                'seeds': List[List[dict{
                    'coord': [x, y]
                    'id_class': id_class,  # cls
                    'score': float,
                    'pred_lane': List[(4+S, ), (4+S, ), (4+S, )]
                }, ...], ...]       外侧List表示batch中不同帧图像，内侧List表示一张图像中不同的预测结果.
            }
        """
        out_seeds = output['seeds']
        results_list = []
        for seeds, img_meta in zip(out_seeds, img_metas):
            seeds = self.nms_seeds_tiny(seeds, self.test_cfg['nms_thr'])    # 其实可以省略nms.
            # seeds: List[dict{
            #             'coord': [x, y]
            #             'id_class': id_class,
            #             'score': float,
            #             'pred_lane': List[(4+S, ), (4+S, ), (4+S, )]
            #             'anchor':
            #             'start_points': (2, )
            #             'theta': (1, )
            #             }, ...]

            if len(seeds) == 0:
                results_list.append([])
                continue

            seeds = sorted(seeds, key=lambda k: k['score'], reverse=True)
            seeds = seeds[:self.test_cfg['max_lanes']]
            scores, labels, lane_preds = self.collect_seeds(seeds)
            # scores = torch.stack(scores, 0)     # (N_pos, )
            # labels = torch.stack(labels, 0)       # (N_pos, )
            # pred_lanes = torch.stack(pred_lanes, 0)   # (N_pos, 2+S)
            pred = self.predictions_to_pred(scores, lane_preds, labels, seeds, img_meta)
            results_list.append(pred)

        return results_list

    def predictions_to_pred(self, scores, lanes_preds, labels, seeds, img_metas=None):
        '''
        Convert predictions to internal Lane structure for evaluation.
        Args:
            scores: (N_pred, )
            lanes_preds: (N_pred, 2+S)
                2+S: 1 start_y(normalized), 1 length(normalized), S coordinates(normalized)
            labels: (N_pred, )
            anchors: (N_pred, 2+S)
        Returns:
            lanes: List[lane0, lane1, ...]
        '''
        ori_height, ori_weight = img_metas['ori_shape'][:2]
        crop = img_metas.get('crop', (0, 0, 0, 0))  # (x_min, y_min, x_max, y_max)
        y_min = crop[1]

        self.prior_ys = self.prior_ys.to(lanes_preds.device)
        self.prior_ys = self.prior_ys.double()
        lanes = []

        for score, lane, label, seed in zip(scores, lanes_preds, labels, seeds):
            # score: float
            # lane: (2+S, )
            lane_xs = lane[2:]  # normalized value
            # start_y(normalized --> absolute)
            start = min(max(0, int(round(lane[0].item() * self.n_strips))),
                        self.n_strips)
            length = int(round(lane[1].item() * self.n_strips))
            end = start + length - 1
            end = min(end, len(self.prior_ys) - 1)

            mask = ~((((lane_xs[:start] >= 0.) & (lane_xs[:start] <= 1.)
                       ).cpu().numpy()[::-1].cumprod()[::-1]).astype(np.bool))
            lane_xs[end + 1:] = -2
            lane_xs[:start][mask] = -2
            lane_ys = self.prior_ys[lane_xs >= 0]  # (N_valid, )   由底向上
            lane_xs = lane_xs[lane_xs >= 0]  # (N_valid, )
            lane_xs = lane_xs.flip(0).double()  # (N_valid, )   由上向底
            lane_ys = lane_ys.flip(0)  # (N_valid, )   由上向底

            lane_ys = (lane_ys * (ori_height - y_min) + y_min) / ori_height
            if len(lane_xs) <= 1:
                continue
            points = torch.stack(
                (lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)),
                dim=1).squeeze(2)  # (N_valid, 2)  normalized

            anchor_ys = (self.prior_ys * (ori_height - y_min) + y_min) / ori_height

            anchor = seed['anchor']
            start_points = seed['start_points']     # (x, y)
            theta = seed['theta']
            lane = Lane(points=points.cpu().numpy(),
                        metadata={
                            'conf': score.cpu().numpy(),
                            'label': label.cpu().numpy(),
                            'anchor': (anchor.cpu().numpy(), anchor_ys.cpu().numpy()),
                            'start_points': start_points.cpu().numpy(),
                            'theta': theta.cpu().numpy()
                        })
            lanes.append(lane)
        return lanes