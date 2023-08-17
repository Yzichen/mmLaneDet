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
from scipy.optimize import linear_sum_assignment


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


@HEADS.register_module
class GANetHead(BaseDenseHead):
    def __init__(self,
                 in_channels=64,
                 num_classes=1,
                 hm_idx=0,
                 loss_heatmap=dict(
                     type='GaussianFocalLoss',
                     alpha=2.0,
                     gamma=4.0,
                     reduction='mean',
                     loss_weight=1.0
                 ),
                 loss_kp_offset=dict(
                     type='L1Loss',
                     reduction='mean',
                     loss_weight=1.0
                 ),
                 loss_sp_offset=dict(
                     type='L1Loss',
                     reduction='mean',
                     loss_weight=0.5
                 ),
                 loss_aux=dict(
                     type='SmoothL1Loss',
                     beta=1.0 / 9.0,
                     reduction='mean',
                     loss_weight=0.2
                 ),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None
                 ):
        super(GANetHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.hm_idx = hm_idx

        self.keypts_head = CtnetHead(
            in_channels=in_channels,
            heads_dict={
                'hm': {'out_channels': 1, 'num_conv': 2},
            },
            final_kernel=1,
            init_bias=-2.19,
            use_bias=True
        )

        self.kp_offset_head = CtnetHead(
            in_channels=in_channels,
            heads_dict={
                'offset': {'out_channels': 2, 'num_conv': 2},
            },
            final_kernel=1,
            init_bias=-2.19,
            use_bias=True
        )

        self.sp_offset_head = CtnetHead(
            in_channels=in_channels,
            heads_dict={
                'offset': {'out_channels': 2, 'num_conv': 2},
            },
            final_kernel=1,
            init_bias=-2.19,
            use_bias=True
        )

        self.loss_heatmap = build_loss(loss_heatmap)
        self.loss_kp_offset = build_loss(loss_kp_offset)
        self.loss_sp_offset = build_loss(loss_sp_offset)
        self.loss_aux = build_loss(loss_aux)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def forward(self, x):
        """
        Args:
            x: dict{
                'features': Tuple((B, C, H3, W3), )      # 经过了dcn, 用于heatmap的预测.
                'aux_feat': (B, C, H3, W3)      # 未经过dcn, 用于回归compensation offset的计算和key_points-start_points之间的offset.
                'deform_points': Tuple((B, num_points*2, H3, W3), )
            }
        Returns:
            dict:{
                kpts_hm: (B, C=1, H3, W3)
                kp_offset: (B, 2, H3, W3)
                sp_offset: (B, 2, H3, W3)
            }
        """
        f_hm = x['features'][0]  # (B, C, H3, W3)
        aux_feat = x['aux_feat']  # (B, C=64, H3, W3)

        # (B, C=64, H3, W3) --> (B, C=64, H3, W3) --> (B, C=1, H3, W3)
        kpts_hm = self.keypts_head(f_hm)['hm']
        kpts_hm = torch.clamp(kpts_hm.sigmoid(), min=1e-4, max=1 - 1e-4)  # (B, C=1, H3, W3)

        if aux_feat is not None:
            f_hm = aux_feat

        # (B, C=64, H3, W3) --> (B, 2, H3, W3)
        kp_offset = self.kp_offset_head(f_hm)['offset']
        # (B, C=64, H3, W3) --> (B, 2, H3, W3)
        sp_offset = self.sp_offset_head(f_hm)['offset']

        output = {
            'kpts_hm': kpts_hm,
            'kp_offset': kp_offset,
            'sp_offset': sp_offset
        }

        return output

    def forward_train(self,
                      x,
                      img_metas,
                      **kwargs):
        outs = self.forward(x)
        # head_dict: {
        #     kpts_hm: (B, C=1, H3, W3)
        #     kp_offset: (B, 2, H3, W3)
        #     sp_offset: (B, 2, H3, W3)
        # }
        deform_points = x['deform_points'][0]      # (B, num_points*2, H3, W3)
        outs['deform_points'] = deform_points

        losses = self.loss(outs, img_metas, **kwargs)

        return losses

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             preds_dicts,
             img_metas,
             gt_targets
             ):
        """Compute losses of the head.
        Args:
            preds_dicts: dict{
                'kpts_hm': (B, C=1, H3, W3)
                'kp_offset': (B, 2, H3, W3)
                'sp_offset': (B, 2, H3, W3)
                'deform_points': (B, num_points*2, H3, W3)
            }
            gt_targets: List[dict(
                'gt_hm_lanes': dict{
                    0: (max_lane_num,  N_sample, 2),
                    1: (max_lane_num,  N_sample, 2),
                    2: (max_lane_num,  N_sample, 2),
                }
                'gt_kpts_hm': (1, hm_H, hm_W)
                'gt_kp_offset': (2, hm_H, hm_W)
                'gt_sp_offset': (2, hm_H, hm_W)
                'kp_offset_mask': (2, hm_H, hm_W)
                'sp_offset_mask': (2, hm_H, hm_W)
            ), ...]
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        kpts_hm = preds_dicts['kpts_hm']    # (B, C=1, H3, W3)
        kp_offset = preds_dicts['kp_offset']    # (B, 2, H3, W3)
        sp_offset = preds_dicts['sp_offset']    # (B, 2, H3, W3)

        gt_kpts_hm = torch.from_numpy(np.stack([target['gt_kpts_hm'] for target in gt_targets], axis=0)).to(
            kpts_hm.device)   # (B, 1, H3, W3)
        gt_kp_offset = torch.from_numpy(np.stack([target['gt_kp_offset'] for target in gt_targets], axis=0)).to(
            kpts_hm.device)   # (B, 2, H3, W3)
        gt_sp_offset = torch.from_numpy(np.stack([target['gt_sp_offset'] for target in gt_targets], axis=0)).to(
            kpts_hm.device)   # (B, 2, H3, W3)
        kp_offset_mask = torch.from_numpy(np.stack([target['kp_offset_mask'] for target in gt_targets], axis=0)).to(
            kpts_hm.device)   # (B, 2, H3, W3)
        sp_offset_mask = torch.from_numpy(np.stack([target['sp_offset_mask'] for target in gt_targets], axis=0)).to(
            kpts_hm.device)   # (B, 2, H3, W3)

        gt_hm_lanes = dict()
        for k in gt_targets[0]['gt_hm_lanes'].keys():
            gt_hm_lanes[k] = torch.from_numpy(np.stack([target['gt_hm_lanes'][k] for target in gt_targets], axis=0)).to(
                kpts_hm.device)     # (B, max_lane_num,  N_sample, 2)
        # 'gt_hm_lanes': dict{
        #    0: (B, max_num_lanes, N_sample0, 2)     2: (y, x)
        #    1: (B, max_num_lanes, N_sample1, 2)
        #    2: (B, max_num_lanes, N_sample2, 2)
        # }

        # 1. heatmap loss
        avg_factor = max(1, gt_kpts_hm.eq(1).sum())
        loss_heatmap = self.loss_heatmap(kpts_hm, gt_kpts_hm, avg_factor=avg_factor)

        # 2. compensation offset loss
        loss_kp_offset = self.loss_kp_offset(
            kp_offset,
            gt_kp_offset,
            weight=kp_offset_mask,
            avg_factor=kp_offset_mask.sum()
        )

        # 3. key-->start offset loss
        loss_sp_offset = self.loss_sp_offset(
            sp_offset,
            gt_sp_offset,
            weight=sp_offset_mask,
            avg_factor=sp_offset_mask.sum()
        )

        # 4. aux loss
        deform_points = preds_dicts['deform_points']    # (B, num_points*2, H3, W3)
        # gt_matched_points: (B, max_num_lanes, N_sample, num_points, 2),
        # pred_matched_points: (B, max_num_lanes, N_sample, num_points, 2)
        gt_matched_points, pred_matched_points = self.assign(
            deform_points,  # (B, num_points*2, H3, W3)
            gt_hm_lanes[self.hm_idx],  # (B, max_num_lanes, N_sample, 2)
        )
        loss_aux = self.loss_aux(
            pred_matched_points,
            gt_matched_points
        )

        loss_dict = dict()
        loss_dict['loss_heatmap'] = loss_heatmap
        loss_dict['loss_kp_offset'] = loss_kp_offset
        loss_dict['loss_sp_offset'] = loss_sp_offset
        loss_dict['loss_aux'] = loss_aux

        return loss_dict

    def assign(self, deform_points, gt_points):
        """
        :param deform_points: (B, num_points*2, H3, W3)   2: (ty, tx)
        :param gt_points: (B, max_num_lanes, N_sample, 2)
        :return:
        """
        def generate_grid(deform_points):
            """
            Args:
                points_map: (B, num_points*2, H3, W3)
            Returns:
                grid: (B, num_points*2, H3, W3)
            """
            batch_size, n_points, fh, fw = deform_points.shape
            # (H3, ) --> (H3, 1, 1) --> (H3, W3, 1)
            y = torch.arange(fh)[:, None, None].repeat(1, fw, 1)
            # (W3, ) --> (1, W3, 1) --> (H3, W3, 1)
            x = torch.arange(fw)[None, :, None].repeat(fh, 1, 1)
            # (H3, W3, 2) --> (1, H3, W3, 2) --> (B, H3, W3, num_points*2)
            coods = torch.cat([y, x], dim=-1)[None, :, :, :].repeat(batch_size, 1, 1, n_points//2).float()
            # (B, H3, W3, num_points*2) --> (B, num_points*2, H3, W3)   2: (y, x)
            grid = coods.permute(0, 3, 1, 2).contiguous().to(deform_points.device)
            return grid

        batch_size, n_points, fh, fw = deform_points.shape
        _, max_num_lanes, n_samples, _ = gt_points.shape
        # generate cood grid
        grid = generate_grid(deform_points)   # (B, num_points*2, H3, W3)
        # get the abs position in feature map, according to gt_points
        # (y, x) + (ty, tx) --> (new_y, new_x)
        # (B, num_points*2, H3, W3) + (B, num_points*2, H3, W3) --> (B, num_points*2, H3, W3)
        points_map = deform_points + grid.contiguous()

        # gt lane points的位置
        gt_points = gt_points.to(points_map.device).contiguous()  # (B, max_num_lanes, N_sample, 2)
        gt_points_int = gt_points.long()
        # (B, max_num_lanes)
        lane_valid_mask = (gt_points_int[:, :, 0, 0] > 0).long()
        # (B, max_num_lanes, N_sample, 2) --> (B, max_num_lanes*N_sample, 2)
        gt_points_int = gt_points_int.reshape(batch_size, -1, 2).contiguous()

        assert n_points % 2 == 0
        n_points = n_points // 2
        # (B, num_points*2, H3, W3) --> (B, num_points, 2, H3, W3)
        points_map = points_map.reshape(batch_size, n_points, 2, fh, fw).contiguous()
        points_cat = []
        # get the point on each lane
        for i in range(batch_size):
            gt_points_int_y = gt_points_int[i, :, 0]  # (max_num_lanes*N_sample, )
            gt_points_int_x = gt_points_int[i, :, 1]  # (max_num_lanes*N_sample, )
            # (num_points, 2, max_num_lanes*N_sample) --> (max_num_lanes*N_sample, num_points, 2)
            # --> (max_num_lanes, N_sample, num_points, 2)
            points_cat.append(
                points_map[i, :, :, gt_points_int_y, gt_points_int_x].permute(2, 0, 1).contiguous().view(
                    max_num_lanes, n_samples, n_points, 2
                )
            )

        points = torch.stack(points_cat, dim=0)   # (B, max_num_lanes, N_sample, num_points, 2)
        points_ = points.unsqueeze(dim=-2)       # (B, max_num_lanes, N_sample, num_points, 1, 2)
        # (B, max_num_lanes, N_sample, 2) --> (B, max_num_lanes, 1, 1, N_sample, 2)
        gt_points_ = gt_points[:, :, None, None, ...].contiguous()
        # compute the distance cost
        # (B, max_num_lanes, 1, 1, N_sample, 2) - (B, max_num_lanes, N_sample, num_points, 1, 2)
        # --> (B, max_num_lanes, N_sample, num_points, N_sample, 2)
        # --> (B, max_num_lanes, N_sample, num_points, N_sample)
        cost = ((gt_points_ - points_) ** 2).sum(-1).detach().cpu()

        # bimatch
        indices = [[[linear_sum_assignment(cost[b_, l_, g_, ...])[1] for g_ in range(n_samples)]
                    for l_ in range(max_num_lanes)] for b_ in range(batch_size)]
        # (B, max_num_lanes, N_sample, num_points)
        indices = np.array(indices)  # 对应的gt indices.

        # align the point bt gt and pred
        # (B, max_num_lanes, N_sample, num_points, 2)
        gt_points_match = torch.cat(
            [torch.cat(
                [gt_points[b_, l_, torch.tensor(indices[b_][l_])].unsqueeze(0) for l_ in range(max_num_lanes)],
                dim=0).unsqueeze(0) for b_ in range(batch_size)],
            dim=0)

        points_match = points
        lane_valid_mask = lane_valid_mask.view(batch_size, max_num_lanes, 1, 1, 1)
        return gt_points_match * lane_valid_mask, points_match * lane_valid_mask

    def simple_test(self, x, img_metas):
        """Test function without test-time augmentation.

        Args:
            x (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n, ).
        """
        outs = self.forward(x)
        # outs: {
        #     kpts_hm: (B, C=1, H3, W3)
        #     kp_offset: (B, 2, H3, W3)
        #     sp_offset: (B, 2, H3, W3)
        # }
        results_list = self.get_lanes(
            outs, img_metas=img_metas)

        return results_list

    @force_fp32(apply_to=('preds_dicts'))
    def get_lanes(self, preds_dicts, img_metas, cfg=None):
        '''
        Convert model output to lanes.
        Args:
            preds_dicts = dict {
                'kpts_hm': (B, C=1, H3, W3)
                'kp_offset': (B, 2, H3, W3)
                'sp_offset': (B, 2, H3, W3)
            }
        Returns:
            results_list: Lane results of each image. The outer list corresponds to each image. The inner list
                corresponds to each lane.
                List[List[lane0, lane1, ...], ...]
        '''
        output_kpts_hms = preds_dicts['kpts_hm']    # (B, C=1, H3, W3)
        output_kp_offsets = preds_dicts['kp_offset']    # (B, 2, H3, W3)
        output_sp_offsets = preds_dicts['sp_offset']    # (B, 2, H3, W3)

        results_list = []
        for kpts_hm, kp_offset, sp_offset, img_meta in zip(output_kpts_hms, output_kp_offsets, output_sp_offsets,
                                                           img_metas):
            # kpts_hm: (C=1, H3, W3)
            # kp_offset: (2, H3, W3)
            # sp_offset: (2, H3, W3)
            kpts_hm = kpts_hm.detach().clone()
            kp_offset = kp_offset.detach().clone()
            sp_offset = sp_offset.detach().clone()

            sp_candidates, valid_kps, valid_sps = self.ktdet_decode(kpts_hm, kp_offset, sp_offset)
            # sp_candidates: (N_start, 2)  2: (x, y)
            # valid_kps: (N_p, 2)     2: (x, y)
            # valid_sps: (N_p, 2)     2: (x, y)

            kpt_groups, groups_start_points = self.group_points(
                sp_candidates,
                valid_kps,
                valid_sps,
            )
            # kpt_groups: List[[(idx, kp, sp), (idx, kp, sp), ...], [(idx, kp, sp), (idx, kp, sp), ...], ...]
            # groups_start_points: List[(2, ), (2, ), ...]     List长度为start points的个数, 每个元素表示start points的(x, y)坐标.

            if len(kpt_groups) == 0:
                results_list.append([])
                continue

            lanes, groups_sps = self.lane_post_process(kpt_groups, groups_start_points)
            # lanes: List[dict{'id_class': int,
            #                  'key_points': List[[x, y], [x, y], ...],
            #                  'start_points':List[[x, y], [x, y], ...]
            #         }, ...]
            # groups_sps: List[dict{'id_class': int, 'group_sp': [x, y]}, ...]
            pred = self.predictions_to_pred(lanes, groups_sps, img_meta)
            results_list.append(pred)

        return results_list

    def ktdet_decode(self, kpts_hm, kp_offset, sp_offset):
        """
        :param kpts_hm: (C=1, H3, W3)
        :param kp_offset: (2, H3, W3)  2: (tx, ty)
        :param sp_offset: (2, H3, W3)  2: (tx, ty)
        :return:
            sp_candidates: (N_start, 2)  2: (x, y)
            valid_kps: (N_p, 2)     2: (x, y)
            valid_sps: (N_p, 2)     2: (x, y)
        """
        def _nms(heat):
            hmax = nn.functional.max_pool2d(heat, (1, 3), stride=(1, 1), padding=(0, 1))
            keep = (hmax == heat).float()       # (B, C=1, H3, W3)
            return heat * keep

        def make_grids(h, w, device):
            """
            Args:
                h: int
                w: int
            Returns:
                coord_mat: (h*w, 2)
            """
            x_coord = torch.arange(0, w, step=1, dtype=torch.float32, device=device)
            y_coord = torch.arange(0, h, step=1, dtype=torch.float32, device=device)
            grid_y, grid_x = torch.meshgrid(y_coord, x_coord)
            grids = torch.stack([grid_x, grid_y], dim=-1)   # (h, w, 2)  2: (x, y)
            grids = grids.view(-1, 2)
            return grids

        fh, fw = kpts_hm.shape[1], kpts_hm.shape[2]
        heat_nms = _nms(kpts_hm.unsqueeze(dim=0)).squeeze(dim=0)    # (C=1, H3, W3)

        # 1. 找到offset<1的点作为候选起始点.
        # Tuple((1, H3, W3),  (1, H3, W3))
        sp_offset_split = torch.split(sp_offset, 1, dim=0)
        mask = torch.lt(sp_offset_split[1], self.test_cfg['root_thr'])   # (1, H3, W3)
        mask_nms = torch.gt(heat_nms, self.test_cfg['kpt_thr'])     # (1, H3, W3)
        mask = mask * mask_nms  # (1, H3, W3)
        mask = mask.squeeze(dim=0)    # (H3, W3)
        sp_candidates = torch.nonzero(mask)   # [[sy0, sx0], [sy1, sx1], ...]    # (N_start, 2)  2: (y, x)
        sp_candidates = sp_candidates[:, [1, 0]]    # (N_start, 2)  2: (x, y)
        sp_candidates = sp_candidates.cpu().numpy()

        heat_nms = heat_nms.permute(1, 2, 0).contiguous().view(-1)      # (H3*W3, )
        kp_offset = kp_offset.permute(1, 2, 0).contiguous().view(-1, 2)     # (H3*W3, 2)
        sp_offset = sp_offset.permute(1, 2, 0).contiguous().view(-1, 2)     # (H3*W3, 2)
        coord_mat = make_grids(fh, fw, device=heat_nms.device)      # (H3*W3, 2)

        kp_mat = coord_mat + kp_offset
        sp_mat = coord_mat + sp_offset
        valid_mat = heat_nms > self.test_cfg['kpt_thr']
        valid_kps = kp_mat[valid_mat].cpu().numpy()   # (N_valid, 2)
        valid_sps = sp_mat[valid_mat].cpu().numpy()   # (N_valid, 2)

        return sp_candidates, valid_kps, valid_sps

    def group_points(self, sp_candidates, kps, sps):
        """
        :param sp_candidates: (N_start, 2)  2: (x, y)
        :param kps: (N_p, 2)     2: (x, y)
        :param sps: (N_p, 2)     2: (x, y)
        :return:
            groups: List[[(idx, kp, sp), (idx, kp, sp), ...], [(idx, kp, sp), (idx, kp, sp), ...], ...]
            groups_start_points: List[(2, ), (2, ), ...]     List长度为start points的个数, 每个元素表示start points的(x, y)坐标.
        """
        def cal_dis(p1, p2):
            result = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            return result

        def choose_mean_point(group):
            """
            Args:
                group: List[(2, ), (2, ), ...]
            Returns:
                mean_point: (2, )
            """
            group_ = np.stack(group, axis=0)    # (N, 2)
            mean_point = np.mean(group_, axis=0)
            # mean_point = mean_point.int()
            return mean_point

        def search_groups(coord, groups, thr):
            """
            Args:
                coord: (2, )
                groups: List[[(2, ), (2, )],  [(2, )],  ...]
            """
            for idx_group, group in enumerate(groups):
                for group_point in group:
                    # group_point: (2, )  array
                    if isinstance(group_point, tuple):
                        group_point_coord = group_point[-1]  # center
                    else:
                        group_point_coord = group_point
                    if cal_dis(coord, group_point_coord) <= thr:
                        return idx_group
            return -1

        def search_groups_by_centers(coord, groups_start_points, by_center_thr):
            """
            Args:
                coord: (2, )  当前key points的(x, y)坐标.
                cluster_centers: List[(2, ), (2, ), ...]     List长度为start points的个数, 每个元素表示start points的(x, y)坐标.
            """
            for idx_group, start_points in enumerate(groups_start_points):
                # idx_group: int
                # cluster_center: (2, )
                dis = cal_dis(coord, start_points)
                if dis <= by_center_thr:
                    return idx_group
            return -1

        def update_coords_fast_by_center(sp_candidates, kps, sps, thr=5, by_center_thr=5):
            groups = []
            groups_sp_candidates = []
            groups_start_points = []

            # group centers first
            # 可能有多个cluster_centers位于同一局部区域, 对应同一个start points.
            # 故先将cluster_centers根据相互之间的距离进行分组，同一组的对应同一个start points.
            for sp_candidate in sp_candidates:
                # sp_candidate: (2, )  2: (x, y)
                idx_group = search_groups(sp_candidate, groups_sp_candidates, thr)
                if idx_group < 0:
                    groups_sp_candidates.append([sp_candidate])
                else:
                    groups_sp_candidates[idx_group].append(sp_candidate)

            # groups_centers: List[[(2,), (2,)], [(2,), ], ...]
            # 外部List长度为start points的个数, 内部List长度为该start points对应sp_candidate的个数.

            # choose mean center
            for group_points in groups_sp_candidates:
                # group_center: List[(2, ), (2, ), ...]
                start_point = choose_mean_point(group_points)  # (2, )
                groups_start_points.append(start_point)
                groups.append([])

            # groups_start_points: List[(2, ), (2, ), ...]   List长度为start points的个数, 每个元素表示start points的(x, y)坐标.
            # groups: List[[], [], ...]     List长度为start points的个数, 内部List暂时为空, 用于保留相应的key points.

            for idx, (kp, sp) in enumerate(zip(kps, sps)):
                # kp: (2, )     预测的key points的(x, y)坐标.
                # sp: (2, )     该key points预测的start points的(x, y)坐标.
                # idx_group为当前key points属于的group.
                idx_group = search_groups_by_centers(sp, groups_start_points, by_center_thr)
                if idx_group < 0:
                    continue
                else:
                    groups[idx_group].append((idx, kp, sp))

            return groups, groups_start_points

        groups, groups_start_points = update_coords_fast_by_center(
            sp_candidates=sp_candidates,
            kps=kps,
            sps=sps,
            thr=self.test_cfg['cluster_by_center_thr'],
            by_center_thr=self.test_cfg['cluster_by_center_thr']
        )

        return groups, groups_start_points

    def lane_post_process(self, kpt_groups, groups_start_points):
        """
        :param kpt_groups: List[[(idx, kp, sp), (idx, kp, sp), ...], [(idx, kp, sp), (idx, kp, sp), ...], ...]
        :param groups_start_points:
                    List[(2, ), (2, ), ...]     List长度为start points的个数, 每个元素表示start points的(x, y)坐标.
        :return:
            lanes: List[dict{'id_class': int,
                             'key_points': List[[x, y], [x, y], ...],
                             'start_points':List[[x, y], [x, y], ...]
                    }, ...]
            groups_sps: List[dict{'id_class': int, 'group_sp': [x, y]}, ...]
        """
        lanes = []
        hm_down_scale = self.test_cfg['hm_down_scale']

        for lane_idx, group in enumerate(kpt_groups):
            # group: List[(idx, kp, sp), (idx, kp, sp), ...]
            # kp: (2, );  sp: (2, )   2: (x, y)
            key_points = []
            start_points = []
            if len(group) > 1:
                for point in group:
                    # point: (idx, kp, sp)
                    key_points.append([point[1][0] * hm_down_scale, point[1][1] * hm_down_scale])
                    start_points.append([point[-1][0] * hm_down_scale, point[-1][1] * hm_down_scale])

                lanes.append(
                    dict(
                        id_class=lane_idx,
                        key_points=key_points,
                        start_points=start_points
                    )
                )
        # lanes: List[dict{'id_class': int,
        #                  'key_points': List[[x, y], [x, y], ...],
        #                  'start_points':List[[x, y], [x, y], ...]
        #             }, ...]

        groups_sps = []
        for group_idx, group_sp in enumerate(groups_start_points):
            group_sp = [group_sp[0] * hm_down_scale, group_sp[1] * hm_down_scale]
            groups_sps.append(
                dict(
                    id_class=group_idx,
                    group_sp=group_sp
                )
            )
        # groups_sps: List[dict{'id_class': int, 'group_sp': [x, y]}, ...]

        return lanes, groups_sps

    def predictions_to_pred(self, lanes, group_sps, img_metas=None):
        '''
        Convert predictions to internal Lane structure for evaluation.
        Args:
            lanes: List[dict{'id_class': int,
                             'key_points': List[[x, y], [x, y], ...],
                             'start_points': List[[x, y], [x, y], ...]
                    }, ...]
            group_sps: List[dict{'id_class': int, 'group_sp': [x, y]}, ...]
        Returns:
            lanes: List[lane0, lane1, ...]
        '''
        ori_height, ori_width = img_metas['ori_shape'][:2]
        img_height, img_width = img_metas['img_shape'][:2]
        crop = img_metas.get('crop', (0, 0, 0, 0))  # (x_min, y_min, x_max, y_max)
        y_min = crop[1]

        ratio_x = ori_width / img_width
        ratio_y = (ori_height - y_min) / img_height

        pred_lanes = []
        for lane_id, lane_dict in enumerate(lanes):
            key_points = lane_dict['key_points']    # List[[x, y], [x, y], ...],
            start_points = lane_dict['start_points']    # List[[x, y], [x, y], ...]
            group_sp = group_sps[lane_id]['group_sp']

            adjusted_key_points = []
            for kp in key_points:
                kp_x = (kp[0] * ratio_x) / ori_width
                kp_y = (kp[1] * ratio_y + y_min) / ori_height
                adjusted_key_points.append((kp_x, kp_y))
            key_points = np.array(adjusted_key_points)      # (N_k, 2)

            adjusted_start_points = []
            for sp in start_points:
                sp_x = (sp[0] * ratio_x) / ori_width
                sp_y = (sp[1] * ratio_y + y_min) / ori_height
                adjusted_start_points.append((sp_x, sp_y))
            start_points = np.array(adjusted_start_points)  # (N_k, 2)

            if len(key_points) <= 2:
                continue

            key_points = sorted(key_points, key=lambda x: x[1])
            filtered_lane = []
            used = set()
            for p in key_points:
                if p[1] not in used:
                    filtered_lane.append(p)
                    used.add(p[1])

            key_points = np.array(filtered_lane)
            lane = Lane(points=key_points,
                        metadata={
                            'start_points': start_points,
                            'group_sp': group_sp
                        })
            pred_lanes.append(lane)
        return pred_lanes

