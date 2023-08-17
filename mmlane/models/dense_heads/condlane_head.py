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
from ..builder import HEADS
from .base_dense_head import BaseDenseHead
from mmlane.core import build_assigner
from mmlane.ops import nms
from mmlane.core import Lane
from mmdet.core.utils import filter_scores_and_topk
from mmcv.cnn import bias_init_with_prob, kaiming_init
import random


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


class DynamicHead(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 num_layers,
                 weight_nums,
                 bias_nums,
                 disable_coords=False,
                 out_channels=1,
                 compute_locations_pre=True,
                 mask_shape=None):
        super(DynamicHead, self).__init__()
        self.num_layers = num_layers
        self.mid_channels = mid_channels
        self.in_channels = in_channels
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.disable_coords = disable_coords
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)
        self.out_channels = out_channels
        self.compute_locations_pre = compute_locations_pre
        self.mask_shape = mask_shape

        if compute_locations_pre and mask_shape is not None:
            H, W = mask_shape
            locations = self.compute_locations(H, W)     # (H, W, 2)
            # (H, W, 2) --> (1, H, W, 2) --> (1, 2, H, W)
            locations = locations.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
            locations[:0, :, :] /= W
            locations[:1, :, :] /= H
            self.locations = locations

    def compute_locations(self, H, W):
        # (H, W), (H, W)
        shift_y, shift_x = torch.meshgrid(torch.arange(0, H, dtype=torch.float32),
                                          torch.arange(0, W, dtype=torch.float32))
        locations = torch.stack((shift_x, shift_y), dim=2)   # (H, W, 2)
        return locations

    def parse_dynamic_params(self,
                             params,
                             mid_channels,
                             weight_nums,
                             bias_nums,
                             out_channels=1,
                             mask=True):
        """
        Args:
            params: (N_pos, num_params=67)    N_pos=n_pos0+n_pos1+...
            mid_channels: branch_out_channels=64
            weight_nums: List[66x1, ]   num_layers=1;     List[66x64, 64x64, 64x1...]  num_layers=3;
            bias_nums:  List[1, ]  num_layers=1;        List[64, 64, 1...]      num_layers=3;
            out_channels: 1
            mask: True
        Returns:
            weight_splits: List[(N_pos, out_channel, in_channel, 1, 1), ]
            bias_splits: List[(N_pos, out_channel), ]
        """
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)

        # List[(N_pos, 66x1), (N_pos, 1)]
        params_splits = list(torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))

        weight_splits = params_splits[:num_layers]  # List[(N_pos, 66), ]
        bias_splits = params_splits[num_layers:]  # List[(N_pos, 1), ]
        if mask:
            bias_splits[-1] = bias_splits[-1] - 2.19

        for l in range(num_layers):
            if l < num_layers - 1:
                # (N_pos, mid_channel*in_channel*1*1) --> (N_pos, mid_channel, in_channel, 1, 1)
                weight_splits[l] = weight_splits[l].view(num_insts, mid_channels, -1, 1, 1)
            else:
                # (N_pos, out_channel*in_channels*1*1) --> (N_pos, out_channel, in_channel, 1, 1)
                weight_splits[l] = weight_splits[l].view(num_insts, out_channels, -1, 1, 1)

        return weight_splits, bias_splits

    def forward(self, x, cond_head_params, num_ins, is_mask=True):
        """
        Args:
            x: (B, C=64, H2, W2)
            cond_head_params: (N_pos, num_params=67)    N_pos=n_pos0+n_pos1+...
            num_ins: List[n_pos0, n_pos1, ...]  len = batch_size
        Returns:
            mask_logits: (1, N_pos*1, H2, W2)
        """
        B, _, H, W = x.size()
        if not self.disable_coords:
            if self.compute_locations_pre and self.mask_shape is not None:
                # (1, 2, H, W) --> (B, 2, H, W)
                locations = self.locations.repeat(B, 1, 1, 1)
                locations = locations.to(x.device)    # (B, 2, H, W)  2: (coord_x, coord_y)
            else:
                locations = self.compute_locations(x.size(2), x.size(3))    # (H, W, 2)
                # (H, W, 2) --> (1, H, W, 2) --> (1, 2, H, W)
                locations = locations.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
                locations[:0, :, :] /= W
                locations[:1, :, :] /= H
                locations = locations.repeat(B, 1, 1, 1)    # (1, 2, H, W) --> (B, 2, H, W)
                locations = locations.to(x.device)

            # Cat[(B, 2, H2, W2), (B, 64, H2, W2)] --> (B, C=66, H2, W2)
            x = torch.cat([locations, x], dim=1)

        dynamic_head_inputs = []
        for batch_id in range(B):
            # (1, C=66, H2, W2) --> (n_pos, C=66, H2, W2)
            dynamic_head_inputs.append(x[batch_id:batch_id + 1, ...].repeat(
                num_ins[batch_id], 1, 1, 1))
        dynamic_head_inputs = torch.cat(dynamic_head_inputs, 0)   # (N_pos, C=66, H2, W2)

        weight_splits, bias_splits = self.parse_dynamic_params(
            cond_head_params,
            self.mid_channels,
            self.weight_nums,
            self.bias_nums,
            out_channels=self.out_channels,
            mask=is_mask)
        # weight_splits: List[(N_pos, out_channel, in_channel, 1, 1), ]
        # bias_splits: List[(N_pos, out_channel), ]
        out_logits = self.dynamic_heads_forward(dynamic_head_inputs, weight_splits,
                                                bias_splits)
        return out_logits

    def dynamic_heads_forward(self, features, weight_splits, bias_splits):
        '''
        :param features: (N_pos, C=66, H2, W2)   C=66
        :param weight_splits: List[(N_pos, out_channel, in_channel, 1, 1), ]
        :param bias_splits: List[(N_pos, out_channel), ]
        :return:
            x: (N_pos, H2, W2)
        '''
        assert features.dim() == 4
        num_insts, C, H, W = features.shape
        x = features.view(1, num_insts*C, H, W)  # (1, N_pos*C, H2, W2)

        n_layers = len(weight_splits)
        for i, (w, b) in enumerate(zip(weight_splits, bias_splits)):
            # w: (N_pos, out_channel, in_channel, 1, 1)
            # b: (N_pos, out_channel)
            w = w.flatten(0, 1)    # (N_pos*out_channel, in_channel, 1, 1)    (out_channels, in_channels/groups, kH, kW)
            b = b.flatten(0, 1)    # (N_pos*out_channel)                      (out_channels, )
            x = F.conv2d(x, w, bias=b, stride=1, padding=0, groups=num_insts)   # (1, N_pos*1, H2, W2)
            if i < n_layers - 1:
                x = F.relu(x)

        return x.squeeze(dim=0)     # (N_pos, H2, W2)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Conv1d(in_channel, out_channel, kernel_size=1)
                                    for in_channel, out_channel in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@HEADS.register_module()
class CondLaneHead(BaseDenseHead):
    def __init__(self,
                 heads_dict={
                     'hm': {'out_channels': 1, 'num_conv': 2},
                 },
                 in_channels=64,
                 num_classes=1,
                 shared_branch_mid_channels=64,
                 shared_branch_out_channels=64,
                 shared_branch_num_conv=3,
                 disable_coords=False,
                 cond_head_num_layers=1,
                 compute_locations_pre=True,
                 mask_shape=(80, 200),
                 with_offset=True,
                 loss_heatmap=dict(
                     type='GaussianFocalLoss',
                     alpha=2.0,
                     gamma=4.0,
                     reduction='mean',
                     loss_weight=1.0
                 ),
                 loss_location=dict(
                     type='SmoothL1Loss',
                     beta=1.0,
                     reduction='mean',
                     loss_weight=1.0
                 ),
                 loss_offset=dict(
                     type='SmoothL1Loss',
                     beta=1.0,
                     reduction='mean',
                     loss_weight=0.4
                 ),
                 loss_range=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     reduction='mean',
                     loss_weight=1.0
                 ),
                 train_cfg=None,
                 test_cfg=dict(
                     hm_thr=0.5,
                     nms_thr=4
                 ),
                 init_cfg=None,
                 ):
        super(CondLaneHead, self).__init__(init_cfg=init_cfg)
        self.heads_dict = heads_dict
        self.in_channels = in_channels
        self.num_classes = num_classes

        # 计算cond_heads所需要的kernel params数量:
        # cond_heads包括：mask head for predict location map;  offset head for predict offset map.
        cond_head_in_channels = shared_branch_out_channels
        self.mask_weight_nums, self.mask_bias_nums = self.cal_num_params(
            in_channels=cond_head_in_channels,
            num_layers=cond_head_num_layers,
            disable_coords=disable_coords,
            out_channels=1
        )
        self.num_mask_params = sum(self.mask_weight_nums) + sum(self.mask_bias_nums)

        self.with_offset = with_offset
        if self.with_offset:
            self.offset_weight_nums, self.offset_bias_nums = self.cal_num_params(
                in_channels=cond_head_in_channels,
                num_layers=cond_head_num_layers,
                disable_coords=disable_coords,
                out_channels=1
            )
            self.num_offset_params = sum(self.offset_weight_nums) + sum(self.offset_bias_nums)

        if self.with_offset:
            self.num_gen_params = self.num_mask_params + self.num_offset_params
        else:
            self.num_gen_params = self.num_mask_params

        # 包含两个head, 一个hm head用于预测heatmap; 一个params head 用于产生对应的kernel params.
        # 该kernel params包含 mask_head(生成location map) 和 offset_head(生成offset map)中的参数.
        if 'params' not in self.heads_dict:
            self.heads_dict['params'] = {
                'out_channels': num_classes * self.num_gen_params,
                'num_conv': 2
            }
        self.ctnet_head = CtnetHead(
            in_channels=in_channels,
            heads_dict=self.heads_dict,
            final_kernel=1,
            init_bias=-2.19,
            use_bias=True)

        # 共享branch
        shared_branch = []
        cur_in_channels = in_channels
        for i in range(shared_branch_num_conv - 1):
            shared_branch.append(
                ConvModule(
                    cur_in_channels,
                    shared_branch_mid_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=dict(type='BN', requires_grad=True)
                )
            )
            cur_in_channels = shared_branch_mid_channels
        shared_branch.append(
            ConvModule(
                shared_branch_mid_channels,
                shared_branch_out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='ReLU')
            )
        )
        self.shared_branch = nn.Sequential(*shared_branch)

        # 构建cond head
        # mask head for location map
        self.mask_head = DynamicHead(
            in_channels=cond_head_in_channels,
            mid_channels=cond_head_in_channels,
            num_layers=cond_head_num_layers,
            weight_nums=self.mask_weight_nums,
            bias_nums=self.mask_bias_nums,
            disable_coords=False,
            out_channels=1,
            compute_locations_pre=compute_locations_pre,
            mask_shape=mask_shape)

        # offset head for offset map
        if self.with_offset:
            self.offset_head = DynamicHead(
                in_channels=cond_head_in_channels,
                mid_channels=cond_head_in_channels,
                num_layers=cond_head_num_layers,
                weight_nums=self.offset_weight_nums,
                bias_nums=self.offset_bias_nums,
                disable_coords=False,
                out_channels=1,
                compute_locations_pre=compute_locations_pre,
                mask_shape=mask_shape
            )

        # 用于预测 vertical range.
        use_sigmoid_cls = loss_range.get('use_sigmoid', False)
        feat_width = mask_shape[-1]     # 200
        self.vertical_range_mlp = MLP(feat_width, hidden_dim=64, output_dim=1 if use_sigmoid_cls else 2, num_layers=2)

        # build loss
        self.loss_heatmap = build_loss(loss_heatmap)
        self.loss_location = build_loss(loss_location)
        self.loss_offset = build_loss(loss_offset)
        self.loss_range = build_loss(loss_range)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def cal_num_params(self,
                       in_channels,
                       num_layers,
                       disable_coords,
                       out_channels=1):
        """
        Args:
            in_channels: 64
            num_layers: int
            disable_coords: bool
            out_channels: 1
        Returns:
            weight_nums: List[]
            bias_nums: List[]
        """
        weight_nums, bias_nums = [], []
        for l in range(num_layers):
            if l == num_layers - 1:
                if num_layers == 1:
                    if not disable_coords:
                        weight_nums.append((in_channels + 2) * out_channels)
                    else:
                        weight_nums.append(in_channels * out_channels)
                else:
                    weight_nums.append(in_channels * out_channels)
                bias_nums.append(out_channels)
            elif l == 0:
                if not disable_coords:
                    weight_nums.append((in_channels + 2) * in_channels)
                else:
                    weight_nums.append(in_channels * in_channels)
                bias_nums.append(in_channels)
            else:
                weight_nums.append(in_channels * in_channels)
                bias_nums.append(in_channels)
        return weight_nums, bias_nums

    def parse_gt_infos(self, batch_gt_infos, hm_shape, mask_shape, device):
        """
        :param gt_infos:
            List[List[dict{
                    'gt_offset': (mask_h, mask_w),  记录offset的gt.
                    'offset_mask': (mask_h, mask_w) 需要监督offset 回归的mask, 只有lane附近的点对应的mask=1
                    'sample_points': List[[x0, y0], [x1, y1], ...], 取这些点对应的kernel params 进行lane shape预测, 提供正确的训练样本.
                    'row': (mask_h), 每一行lane对应的准确的pos_x (float).
                    'row_mask': (mask_h, ), 表示location预测的监督mask.
                    'range': (mask_h),  表明每一行是否存在车道线.
                    'label': 0
                }, ...], ...],   外层List表示不同batch, 内层List表示不同车道.
        :return:
            poses: (N_pos, )
            gt_targets = {
                'rows': (N_pos, mask_h)
                'row_masks': (N_pos, mask_h)
                'gt_offsets': (N_pos, mask_h, mask_w)
                'offsets_masks': (N_pos, mask_h, mask_w)
                'lane_ranges': (N_pos, mask_h)
            }
        """
        n_cls = self.num_classes
        hm_h, hm_w = hm_shape[:2]
        mask_h, mask_w = mask_shape[:2]

        poses = []
        gt_offsets = []
        offsets_masks = []
        rows = []
        row_masks = []
        lane_ranges = []
        num_ins = []
        for batch_idx, img_infos in enumerate(batch_gt_infos):
            # img_infos: 对应某张图像中所有车道的gt信息.
            num = 0
            for lane_info in img_infos:
                # lane_info: 对应某条车道的gt信息.
                # lane_info: dict{
                #     'gt_offset': (mask_h, mask_w),  记录offset的gt.
                #     'offset_mask': (mask_h, mask_w) 需要监督offset 回归的mask, 只有lane附近的点对应的mask=1
                #     'sample_points': List[[x0, y0], [x1, y1], ...], 取这些点对应的kernel params 进行lane shape预测, 提供正确的训练样本.
                #     'row': (mask_h), 每一行lane对应的准确的pos_x (float).
                #     'row_mask': (mask_h, ), 表示location预测的监督mask.
                #     'range': (mask_h),  表明每一行是否存在车道线.
                #     'label': 0
                # }
                label = lane_info['label']
                num += len(lane_info['sample_points'])      # 用于lane shape预测的训练样本数量.

                gt_offset = torch.from_numpy(lane_info['gt_offset']).to(device)     # (mask_h, mask_w)
                offset_mask = torch.from_numpy(lane_info['offset_mask']).to(device)     # (mask_h, mask_w)
                row = torch.from_numpy(lane_info['row']).to(device)     # (mask_h, )
                row_mask = torch.from_numpy(lane_info['row_mask']).to(device)   # (mask_h, )
                lane_range = torch.from_numpy(lane_info['range']).to(device)    # (mask_h, )

                # 从heat_map(B, C=1, hm_H, hm_W)中取对应点位置[x, y]的kernel params, 进行lane shape预测, 提供正确的训练样本.
                # 这里记录对应点在heat map中的索引.
                for p in lane_info['sample_points']:
                    pos = batch_idx * n_cls * hm_h * hm_w + label * hm_h * hm_w + \
                          p[1] * hm_w + p[0]
                    poses.append(pos)

                for i in range(len(lane_info['sample_points'])):
                    rows.append(row)
                    row_masks.append(row_mask)
                    gt_offsets.append(gt_offset)
                    offsets_masks.append(offset_mask)
                    lane_ranges.append(lane_range)

            if num == 0:
                # 如果该图像中没有训练样本点（也就是没有车道线),  在图像中随机选取一个.
                # 但应该只有lane的vertical range会得到监督.
                gt_offset = torch.zeros((mask_h, mask_w)).to(device)
                offset_mask = torch.zeros((mask_h, mask_w)).to(device)
                row = torch.zeros((mask_h, )).to(device)
                row_mask = torch.zeros((mask_h, )).to(device)
                lane_range = torch.zeros((mask_h, ), dtype=torch.int64).to(device)

                label = 0
                pos = batch_idx * n_cls * hm_h * hm_w + random.randint(0, n_cls * hm_h * hm_w - 1)
                num = 1
                poses.append(pos)
                gt_offsets.append(gt_offset)
                offsets_masks.append(offset_mask)
                rows.append(row)
                row_masks.append(row_mask)
                lane_ranges.append(lane_range)

            num_ins.append(num)

        if len(poses) > 0:
            poses = torch.from_numpy(np.array(poses, dtype=np.long)).to(device)
            rows = torch.stack(rows, dim=0)     # (N_pos, mask_h)
            row_masks = torch.stack(row_masks, dim=0)       # (N_pos, mask_h)
            gt_offsets = torch.stack(gt_offsets, dim=0)      # (N_pos, mask_h, mask_w)
            offsets_masks = torch.stack(offsets_masks, dim=0)   # (N_pos, mask_h, mask_w)
            lane_ranges = torch.stack(lane_ranges, dim=0)   # (N_pos, mask_h)

        gt_targets = {
            'rows': rows,       # (N_pos, mask_h)
            'row_masks': row_masks,     # (N_pos, mask_h)
            'gt_offsets': gt_offsets,   # (N_pos, mask_h, mask_w)
            'offsets_masks': offsets_masks,     # (N_pos, mask_h, mask_w)
            'lane_ranges': lane_ranges  # (N_pos, mask_h)
        }

        return poses, num_ins, gt_targets

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
        batch_gt_infos = kwargs['gt_masks']
        if len(x) == 2:
            h_mask, w_mask = x[0].shape[2:]
            h_hm, w_hm = x[1].shape[2:]
        else:
            h_mask, w_mask = x[0].shape[2:]
            h_hm, w_hm = x[0].shape[2:]
        device = x[0].device
        poses, num_ins, gt_mask_targets = self.parse_gt_infos(batch_gt_infos, hm_shape=(h_hm, w_hm),
                                                mask_shape=(h_mask, w_mask), device=device)

        outs = self(x, poses, num_ins)
        losses = self.loss(outs, img_metas, gt_mask_targets, **kwargs)
        return losses

    def simple_test(self, feats, img_metas):
        """Test function without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n, ).
        """
        outs = self(feats)
        results_list = self.get_lanes(
            outs, img_metas=img_metas)

        return results_list

    def forward(self, x, poses=None, num_ins=None):
        # f_mask: 用于预测 lane shape 的feature map.     # (B, C=64, H2, W2)
        # f_hm: 用于预测 heatmap 和 kernel params 的feature map.  # (B, C=64, H4, W4)
        if len(x) == 2:
            f_mask, f_hm = x[0], x[1]
        else:
            f_mask, f_hm = x[0], x[0]
        batch_size = f_mask.shape[0]
        h_hm, w_hm = f_hm.shape[2:]
        h_mask, w_mask = f_mask.shape[2:]

        result = self.ctnet_head(f_hm)
        # hm: (B, 1, H4, W4)
        # params: (B, 134, H4, W4)   134: (64+2)*1+1 + (64+2)*1+1
        hms, params = result['hm'], result['params']
        # (B, 134, H4, W4) --> (B, n_cls=1, 134, H4, W4)
        params = params.view(batch_size, self.num_classes, -1, h_hm, w_hm)

        # (B, C=64, H2, W2) --> (B, C=64, H2, W2)
        shared_features = self.shared_branch(f_mask)

        if self.training:
            sample_points = poses.unsqueeze(dim=-1)     # (N_pos, 1)
            sample_tensor = sample_points.repeat([1, self.num_gen_params])
            mask_sample_tensor = sample_tensor[:, :self.num_mask_params]

            # (B, n_cls=1, 134, H4, W4) --> (B, n_cls, H4, W4, num_params=134) --> (B*n_cls*H4*W4, num_params=134)
            params = params.permute(0, 1, 3, 4,
                                    2).contiguous().view(-1, self.num_gen_params)
            # 预测lane location map.
            # (B*n_cls*H4*W4, num_params=67) --> (N_pos, num_params=67)
            mask_params = params[:, :self.num_mask_params].gather(
                0, mask_sample_tensor)
            masks = self.mask_head(shared_features, mask_params, num_ins)  # (N_pos, H2, W2)

            # 预测lane offset map.
            if self.with_offset:
                offset_sample_tensor = sample_tensor[:, self.num_mask_params:]
                offset_params = params[:, self.num_mask_params:].gather(
                    0, offset_sample_tensor)
                offsets = self.offset_head(shared_features, offset_params, num_ins)  # (N_pos, H2, W2)
            else:
                offsets = None

            # (N_pos, H2, W2) --> (N_pos, W2, H2)
            range_feature = masks.permute(0, 2, 1).contiguous()
            # (N_pos, W2, H2) --> (N_pos, 64, H2) --> (N_pos, 2, H2)
            vertical_range = self.vertical_range_mlp(range_feature)

            output = {
                'hm': hms.sigmoid(),     # (B, 1, H4, W4)
                'masks': masks,         # (N_pos, H2, W2)
                'offsets': offsets,     # (N_pos, H2, W2)
                'vertical_range': vertical_range,   # (N_pos, 2, H2)
            }
            return output
        else:
            # (B, n_cls=1, 134, H4, W4) --> (B, n_cls, H4, W4, num_params=134) --> (B, n_cls*H4*W4, num_params=134)
            params = params.permute(0, 1, 3, 4,
                                    2).contiguous().view(batch_size, -1, self.num_gen_params)
            hms = torch.clamp(hms.sigmoid(), min=1e-4, max=1 - 1e-4)  # (B, 1, H4, W4)
            out_seeds = []
            out_hms = []
            for hm, param, shared_feature in zip(hms, params, shared_features):
                """
                    hm: (n_cls(1), h_hm, w_hm)
                    param: (n_cls(1)*h_hm*w_hm, num_params(134))
                    shared_feature: (C=64, h_mask, w_mask) 
                """
                shared_feature = shared_feature.unsqueeze(dim=0)  # (1, C=64, h_mask, w_mask)
                # 从heatmap利用max_pool，找到positive points, 记录对应的坐标，类别.
                seeds = self.ctdet_decode(hm, thr=self.test_cfg['hm_thr'])
                # seeds: List[dict{
                #     'coord': coord,  # [x, y]
                #     'id_class': id_class,   # cls+1
                #     'score': score
                # }, ...]     # len = n_pos
                if len(seeds):
                    pose = self.parse_pos(seeds, (hm.shape[1], hm.shape[2]))
                    pose = torch.tensor(pose, dtype=torch.long, device=hm.device)
                    num_ins = [pose.shape[0]]
                    sample_points = pose.unsqueeze(dim=-1)  # (n_pos, 1)

                    sample_tensor = sample_points.repeat([1, self.num_gen_params])
                    mask_sample_tensor = sample_tensor[:, :self.num_mask_params]
                    # 预测lane location map.
                    # (n_cls*H4*W4, num_params=67) --> (n_pos, num_params=67)
                    mask_params = param[:, :self.num_mask_params].gather(0, mask_sample_tensor)
                    masks = self.mask_head(shared_feature, mask_params, num_ins)  # (n_pos, H2, W2)

                    # 预测lane offset map.
                    if self.with_offset:
                        offset_sample_tensor = sample_tensor[:, self.num_mask_params:]
                        offset_params = param[:, self.num_mask_params:].gather(
                            0, offset_sample_tensor)
                        offsets = self.offset_head(shared_feature, offset_params, num_ins)  # (n_pos, H2, W2)
                    else:
                        offsets = None

                    # (n_pos, H2, W2) --> (n_pos, W2, H2)
                    range_feature = masks.permute(0, 2, 1).contiguous()
                    # (n_pos, W2, H2) --> (n_pos, 64, H2) --> (n_pos, 2, H2)
                    vertical_range = self.vertical_range_mlp(range_feature)

                    for seed_id in range(len(seeds)):
                        seeds[seed_id]['mask'] = masks[seed_id]    # (H2, W2)
                        seeds[seed_id]['range'] = vertical_range[seed_id]   # (2, H2)
                        if offsets is not None:
                            seeds[seed_id]['offset'] = offsets[seed_id]  # (H2, W2)
                        else:
                            seeds[seed_id]['offset'] = masks.new_zeros((masks.shape[1], masks.shape[2]))
                else:
                    seeds = []

                out_seeds.append(seeds)
                out_hms.append(hm)

            output = {'seeds': out_seeds, 'hm': out_hms}
            return output

    def loss(self, preds_dicts, img_metas, gt_mask_targets, gt_hm, **kwargs):
        """
        :param preds_dicts: dict{
                                'hm': hm.sigmoid(),     # (B, 1, H4, W4)
                                'masks': masks,     # (N_pos, H2, W2)
                                'offsets': offsets,     # (N_pos, H2, W2)
                                'vertical_range': vertical_range,   # (N_pos, 2, H2)
                            }
        :param img_metas:
        :param gt_mask_targets: dict{
                                'rows': rows,       # (N_pos, mask_h)
                                'row_masks': row_masks,     # (N_pos, mask_h)
                                'gt_offsets': gt_offsets,   # (N_pos, mask_h, mask_w)
                                'offsets_masks': offsets_masks,     # (N_pos, mask_h, mask_w)
                                'lane_ranges': lane_ranges  # (N_pos, mask_h)
                            }
        :param gt_hm: (B, 1, hm_h, hm_w)
        :return:
        """
        # compute heatmap loss
        hm = preds_dicts['hm']          # (B, 1, H4, W4)
        avg_factor = max(1, gt_hm.eq(1).sum())
        loss_heatmap = self.loss_heatmap(hm, gt_hm, avg_factor=avg_factor)

        # compute location loss
        masks = preds_dicts['masks']    # (N_pos, H2, W2)
        mask_softmax = F.softmax(masks, dim=-1)     # (N_pos, H2, W2)
        x_pos = torch.arange(0, mask_softmax.shape[-1], step=1, dtype=torch.float32, device=mask_softmax.device)
        # (W2, ) --> (N_pos, H2, W2)
        x_pos = x_pos.view(1, 1, x_pos.shape[0]).repeat(mask_softmax.shape[0], mask_softmax.shape[1], 1)
        row_pos = torch.sum(mask_softmax * x_pos, dim=-1) + 0.5      # (N_pos, H2)

        gt_row_pos = gt_mask_targets['rows']     # (N_pos, H2)
        row_masks = gt_mask_targets['row_masks']    # (N_pos, H2)
        loss_location = self.loss_location(row_pos, gt_row_pos, weight=row_masks, avg_factor=row_masks.sum())

        # compute offset loss
        if self.with_offset:
            pred_offsets = preds_dicts['offsets']    # (N_pos, H2, W2)
            gt_offsets = gt_mask_targets['gt_offsets']      # (N_pos, H2, W2)
            offsets_masks = gt_mask_targets['offsets_masks']    # (N_pos, H2, W2)
            loss_offset = self.loss_offset(pred_offsets, gt_offsets, weight=offsets_masks,
                                           avg_factor=offsets_masks.sum())

        # compute range loss
        pred_range = preds_dicts['vertical_range']      # (N_pos, 2, H2)
        gt_ranges = gt_mask_targets['lane_ranges']      # (N_pos, H2)
        loss_range = self.loss_range(pred_range, gt_ranges)

        loss_dict = {
            'loss_heatmap': loss_heatmap,
            'loss_location': loss_location,
            'loss_range': loss_range
        }
        if self.with_offset:
            loss_dict['loss_offset'] = loss_offset

        return loss_dict

    def ctdet_decode(self, heat, thr=0.1):
        """
        Args:
            heat: (n_cls(1), h_hm, w_hm)
        Returns:
            seeds: List[dict{
                'coord': coord,  # [x, y]
                'id_class': id_class,   # cls+1
                'score': score
            }, ...]     # len = N_pos
        """
        def _nms(heat, kernel=3):
            pad = (kernel - 1) // 2
            hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
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
                id_class = c + 1
                coord = [x, y]
                score = heat[y, x, c]
                ret.append({
                    'coord': coord,  # [x, y]
                    'id_class': id_class,   # cls+1
                    'score': score
                })
            return ret

        # (n_cls(1), h_hm, w_hm)
        heat_nms = _nms(heat)
        # (n_cls(1), h_hm, w_hm) --> (h_hm, w_hm, n_cls(1))
        heat_nms = heat_nms.permute(1, 2, 0).contiguous()
        # Tuple(h_id(n_pos, ), w_id(n_pos, ),  cls_id(n_pos, ))
        inds = torch.where(heat_nms > thr)
        seeds = _format(heat_nms, inds)
        return seeds

    def parse_pos(self, seeds, hm_shape):
        """
        Args:
            seeds: List[dict{
                        'coord': coord,  # [x, y]
                        'id_class': id_class,   # cls+1
                        'score': score
                    }, ...]     # len = N_pos
            hm_shape: (h, w)
        Returns:
            poses:  (N_pos, )
        """
        hm_h, hm_w = hm_shape
        pose = []
        for seed in seeds:
            x, y = seed['coord']
            label = seed['id_class'] - 1
            pos = label * hm_h * hm_w + y * hm_w + x
            pose.append(pos)
        return pose

    def get_lanes(self, output, img_metas):
        """
        Args:
            output: dict
            {
                'seeds': List[List[dict{
                    'coord': [x, y]
                    'id_class': id_class,  # cls+1
                    'score': float,
                    'mask': (H2, W2),
                    'offset': (H2, W2),
                    'range': (2, H2)
                }, ...], ...]       外侧List表示batch中不同帧图像，内侧List表示一张图像中不同的预测结果.

                'hm': List[(n_cls(1), h_hm, w_hm), (n_cls(1), h_hm, w_hm), ...]
            }
        """
        out_seeds, out_hm = output['seeds'], output['hm']
        results_list = []
        for seeds, hm, img_meta in zip(out_seeds, out_hm, img_metas):
            # hm: (n_cls(1), h_hm, w_hm)

            # 此时seed点中可能仍包含一些距离较近的点，即代表同一条车道，因此利用score和距离阈值thr进行过滤.
            seeds = self.nms_seeds_tiny(seeds, self.test_cfg['nms_thr'])
            # seeds: List[dict{
            #             'coord': [x, y]
            #             'id_class': id_class,  # cls+1
            #             'score': float,
            #             'mask': (H2, W2)
            #             'offset': (H2, W2),
            #             'range': (2, H2)
            #             }, ...]

            if len(seeds) == 0:
                results_list.append([])
                continue

            masks, offsets, ranges, scores = self.collect_seeds(seeds)
            # masks: (n_pos, H2, W2)
            # offset: (n_pos, H2, W2)
            # ranges: (n_pos, 2, H2)
            # scores: List[float, float, ...]
            pred = self.predictions_to_pred(masks, offsets, ranges, scores, img_meta)
            results_list.append(pred)

        return results_list

    def predictions_to_pred(self, masks, offsets, ranges, scores, img_meta):
        """
        :param masks: (n_pos, H2, W2)
        :param offsets: (n_pos, H2, W2)
        :param ranges: (n_pos, 2, H2)
        :param scores: List[float, float, ...]
        :param img_meta:
        :return:
        """
        def get_range(ranges):
            """
            Args:
                ranges: (N_pos, H2)
            Returns:
                lane_ends: List[[min_idx, max_idx], [min_idx, max_idx], ...]
            """
            max_rows = ranges.shape[1]
            lane_ends = []
            for idx, lane_range in enumerate(ranges):
                # lane_range: (H2, )
                min_idx = max_idx = None
                for row_idx, valid in enumerate(lane_range):
                    if valid:
                        min_idx = row_idx - 1
                        break
                for row_idx, valid in enumerate(lane_range[::-1]):
                    if valid:
                        max_idx = len(lane_range) - row_idx
                        break
                if max_idx is not None:
                    max_idx = min(max_rows - 1, max_idx)
                if min_idx is not None:
                    min_idx = max(0, min_idx)
                lane_ends.append([min_idx, max_idx])
            return lane_ends

        down_scale = self.test_cfg['down_scale']
        ori_height, ori_weight = img_meta['ori_shape'][:2]
        img_h, img_w = img_meta['img_shape'][:2]
        crop = img_meta.get('crop', (0, 0, 0, 0))  # (x_min, y_min, x_max, y_max)
        y_min = crop[1]

        # 根据masks计算车道在每行对应的位置row_pos.
        mask_softmax = F.softmax(masks, dim=-1)  # (n_pos, H2, W2) --> (n_pos, H2, W2)
        x_pos = torch.arange(0, mask_softmax.shape[-1], step=1, dtype=torch.float32, device=mask_softmax.device)
        # (W2, ) --> (n_pos, H2, W2)
        x_pos = x_pos.view(1, 1, x_pos.shape[0]).repeat(mask_softmax.shape[0], mask_softmax.shape[1], 1)
        row_pos = torch.sum(mask_softmax * x_pos, dim=-1).detach().cpu().numpy().astype(np.int32)   # (n_pos, H2)
        # 获得offsets
        offsets = offsets.detach().cpu().numpy()  # (N_pos, H2, W2)
        # 获得预测车道的范围
        ranges = torch.argmax(ranges, 1).detach().cpu().numpy()     # (N_pos, H2)
        lane_ranges = get_range(ranges)     # List[[min_y, max_y], [min_y, max_y], ...]

        num_lanes = len(masks)
        lanes = []
        for lane_idx in range(num_lanes):
            if lane_ranges[lane_idx][0] is None or lane_ranges[lane_idx][1] is None:
                continue

            selected_ys = np.arange(lane_ranges[lane_idx][0],
                                    lane_ranges[lane_idx][1] + 1)  # (n_selected_rows, )
            if len(selected_ys) <= 1:
                continue

            cur_row_pos = row_pos[lane_idx]     # (H2, )
            selected_xs = cur_row_pos[selected_ys]      # (n_selected_rows, )
            if self.with_offset:
                selected_offsets = offsets[lane_idx, selected_ys, selected_xs]    # (n_selected_rows, )
            else:
                selected_offsets = 0.5
            lane_points = np.stack((selected_xs, selected_ys), axis=1).astype(np.float32)   # (n_selected_rows, 2)
            lane_points[:, 0] += selected_offsets
            lane_points *= down_scale

            lane_xs = lane_points[:, 0]
            lane_ys = lane_points[:, 1]
            ratio_y = (ori_height - y_min) / img_h
            ratio_x = ori_weight / img_w

            lane_ys = (lane_ys * ratio_y + y_min) / ori_height
            lane_xs = (lane_xs * ratio_x) / ori_weight

            lane_points = np.stack((lane_xs, lane_ys), axis=1)    # (n_points, 2)
            lane = Lane(points=lane_points,
                        metadata={
                            'conf': scores[lane_idx],
                            'label': 0
                        })
            lanes.append(lane)

        return lanes

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
            return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

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
                'mask': (H2, W2),
                'offset': (H2, W2),
                'range': (1, 2, H2),
            }, ...]
        Returns:
            masks: (N_pos, H2, W2)
            offsets:  (N_pos, H2, W2)
            scores: List[float, float, ...]
            ranges: (N_pos, 2, H2)
        """
        masks = []
        offsets = []
        scores = []
        ranges = []
        for seed in seeds:
            masks.append(seed['mask'])
            offsets.append(seed['offset'])
            scores.append(seed['score'])
            ranges.append(seed['range'])
        if len(masks) > 0:
            masks = torch.stack(masks, 0)     # (N_pos, H2, W2)
            offsets = torch.stack(offsets, 0)       # (N_pos, H2, W2)
            ranges = torch.stack(ranges, 0)   # (N_pos, 2, W2)
            return masks, offsets, ranges, scores
        else:
            return None

