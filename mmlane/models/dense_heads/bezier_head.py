import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import ConvModule
from ..builder import HEADS
from .base_dense_head import BaseDenseHead
from mmlane.models import build_loss
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean
from mmlane.core import build_assigner
from mmlane.core.lane import BezierCurve
from mmlane.core import Lane


class ConvProjection_1D(torch.nn.Module):
    def __init__(self, num_layers, in_channels, bias=True, k=3):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_layers = nn.ModuleList(nn.Conv1d(in_channels if i > 0 else in_channels,
                                                     in_channels, kernel_size=k, bias=bias, padding=(k - 1) // 2)
                                           for i in range(num_layers))
        self.hidden_norms = nn.ModuleList(nn.BatchNorm1d(in_channels) for _ in range(num_layers))

    def forward(self, x):
        """
        Args:
            x: (B, C=256, W=40)
        Returns:
            x: (B, C=256, W=40)
        """
        for conv, norm in zip(self.hidden_layers, self.hidden_norms):
            x = F.relu(norm(conv(x)))

        return x


class SimpleSegHead(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes):
        super(SimpleSegHead, self).__init__()
        self.conv = ConvModule(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.conv_out = nn.Conv2d(mid_channels, num_classes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x


@HEADS.register_module()
class BezierHead(BaseDenseHead):
    def __init__(self,
                 in_channels=256,
                 branch_channels=256,
                 num_proj_layers=2,
                 feature_size=(23, 40),
                 order=3,
                 num_sample_points=100,
                 with_seg=True,
                 num_classes=1,
                 seg_num_classes=1,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     class_weight=1.0/0.4,
                     reduction='mean',
                     loss_weight=0.1
                 ),
                 loss_reg=dict(
                     type='L1Loss',
                     reduction='mean',
                     loss_weight=1.0,
                 ),
                 loss_seg=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     ignore_index=255,
                     class_weight=1.0/0.4,
                     reduction='mean',
                     loss_weight=0.75
                 ),
                 sync_cls_avg_factor=True,
                 train_cfg=dict(
                     assigner=dict(
                         type='BezierHungarianAssigner',
                     )
                 ),
                 test_cfg=None,
                 init_cfg=None,
                 ):
        super(BezierHead, self).__init__(init_cfg)
        h, w = feature_size
        self.num_classes = num_classes
        self.bezier_curve = BezierCurve(order=order)
        self.num_sample_points = num_sample_points

        self.avg_pool = nn.AvgPool2d(kernel_size=(h, 1), stride=1, padding=0)
        shared_branchs = []
        for i in range(num_proj_layers):
            branch = ConvModule(
                in_channels=in_channels,
                out_channels=in_channels if i < num_proj_layers - 1 else branch_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=dict(type='Conv1d'),
                norm_cfg=dict(type='BN1d'),
                act_cfg=dict(type='ReLU')
            )
            shared_branchs.append(branch)
        self.shared_branchs = nn.Sequential(*shared_branchs)

        self.cls_layer = nn.Conv1d(in_channels=branch_channels,
                                   out_channels=num_classes,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   bias=True,
                                   )

        self.num_control_points = order + 1
        self.reg_layer = nn.Conv1d(in_channels=branch_channels,
                                   out_channels=self.num_control_points * 2,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   bias=True
                                   )

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is BezierHead):
           self.cls_pos_weight = class_weight

        self.loss_cls = build_loss(loss_cls)
        self.loss_reg = build_loss(loss_reg)

        self.with_seg = with_seg
        if self.with_seg:
            self.use_sigmoid_seg = loss_seg.get('use_sigmoid', False)
            if self.use_sigmoid_seg:
                self.seg_out_channels = seg_num_classes
            else:
                self.seg_out_channels = seg_num_classes + 1

            self.seg_decoder = SimpleSegHead(in_channels=in_channels,
                                             mid_channels=in_channels,
                                             num_classes=self.seg_out_channels)

            class_weight = loss_seg.get('class_weight', None)
            if class_weight is not None:
                self.seg_pos_weight = class_weight
            self.loss_seg = build_loss(loss_seg)
            self.seg_ignore_index = loss_seg.get('ignore_index', 255)

        if train_cfg:
            assigner = train_cfg['assigner']
            self.assigner = build_assigner(assigner)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def forward_train(self,
                      x,
                      f,
                      img_metas,
                      **kwargs):
        """
        Args:
            x: (B, C, H, W)   经过了dilated_blocks 和 flip_fusion
            f: (B, C, H, W)   未经过dilated_blocks 和 flip_fusion
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x, f)
        losses = self.loss(outs, img_metas, **kwargs)
        return losses

    def forward(self, x, f=None):
        """
        :param x: (B, C, H, W), 经过了dilated_blocks 和 flip_fusion.
        :param f: (B, C, H, W), 未经过dilated_blocks 和 flip_fusion
        :return:
        """
        x = self.avg_pool(x).squeeze(dim=2)      # (B, C, W)
        batch_size, _, n_q = x.shape

        x = self.shared_branchs(x)   # (B, C, W)
        # (B, C, W) --> (B, 1, W)
        logits = self.cls_layer(x)
        logits = logits.permute(0, 2, 1).contiguous()   # (B, W, 1)
        # (B, C, W) --> (B, 4*2, W)
        pred_control_points = self.reg_layer(x)
        # (B, 4*2, W) --> (B, W, 4*2) --> (B, W, 4, 2)
        pred_control_points = pred_control_points.permute(0, 2, 1).contiguous().\
            view(batch_size, n_q, self.num_control_points, 2)

        output = {
            'logits': logits,    # (B, W, 1)
            'pred_control_points': pred_control_points,   # (B, W, 4, 2)
        }
        if self.training:
            if self.with_seg:
                seg = self.seg_decoder(f)   # (B, 1, H, W)
                output['pred_seg'] = seg

        return output

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             preds_dicts,
             img_metas,
             gt_control_points,
             gt_labels,
             gt_semantic_seg=None,
             ):
        """
        :param preds_dicts: dict{
                                'logits': (B, W, 1)
                                'pred_control_points': (B, W, 4, 2)
                                'seg': (B, 1, H, W)
                            }
        :param img_metas:
        :param gt_control_points:  List[(N_gt0, 4, 2), (N_gt1, 4, 2), ...]
        :param gt_labels:   List[(N_gt0,), (N_gt1,), ...]
        :param gt_semantic_seg: (B, 1, img_H, img_W)
        :return:
        """

        cls_logits = preds_dicts['logits']  # (B, N_q, 1)
        pred_control_points = preds_dicts['pred_control_points']    # (B, N_q, 4, 2)

        num_imgs = cls_logits.shape[0]
        # List[(N_q, 1), (N_q, 1), ...]
        cls_logits_list = [cls_logits[i] for i in range(num_imgs)]
        # List[(N_q, 4, 2), (N_q, 4, 2), ...]
        pred_control_points_list = [pred_control_points[i] for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(
            cls_logits_list,    # List[(N_q, 1), (N_q, 1), ...]    len = batch_size
            pred_control_points_list,    # List[(N_q, 4, 2), (N_q, 4, 2), ...]  len = batch_size
            gt_control_points,       # List[(N_gt0, 4, 2), (N_gt1, 4, 2), ...]     len = batch_size
            gt_labels       # List[(N_gt0,), (N_gt1,), ...]
        )
        # cls_reg_targets:
        # labels_list: List[(N_q, ), (N_q, ), ...]
        # label_weights_list: List[(N_q, ), (N_q, ), ...]
        # control_points_targets_list: List[(N_q, 4, 2), (N_q, 4, 2), ...]
        # control_points_weights_list: List[(N_q, ), (N_q, ), ...]
        # pos_inds_list: int    N_pos0+N_pos1+...
        # neg_inds_list: int    N_neg0+N_neg1+...

        (labels_list, label_weights_list, control_points_targets_list, control_points_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)      # (B*N_q, )
        label_weights = torch.cat(label_weights_list, 0)    # (B*N_q, )
        control_points_targets = torch.cat(control_points_targets_list, 0)      # (B*N_q, 4, 2)
        control_points_weights = torch.cat(control_points_weights_list, 0)      # (B*N_q, )

        loss_dict = dict()
        # 1. classification loss  bce loss
        cls_scores = cls_logits.view(-1, cls_logits.shape[-1])      # (B*N_q, n_cls=1)
        loss_cls = self.loss_cls(
            cls_scores,      # (B*num_priors, n_cls)
            labels,          # (B*num_priors, )
            label_weights * 1/self.cls_pos_weight,   # (B*num_priors, )
            )
        loss_dict['loss_cls'] = loss_cls

        # 2. regression loss
        pred_control_points = pred_control_points.view(-1, self.num_control_points, 2)  # (B*N_q, 4, 2)
        gt_control_points = control_points_targets
        # (B*N_q, N_sample_points=100, 2)
        pred_sample_points = self.bezier_curve.get_sample_points(control_points_matrix=pred_control_points,
                                                                 num_sample_points=self.num_sample_points)
        # (B*N_q, N_sample_points=100, 2)
        gt_sample_points = self.bezier_curve.get_sample_points(control_points_matrix=gt_control_points,
                                                               num_sample_points=self.num_sample_points)
        reg_weight = control_points_weights.view(control_points_weights.shape[0], 1, 1).\
            repeat(1, self.num_sample_points, 2)    # (B*N_q, N_sample_points=100, 2)
        avg_factor = num_total_pos * self.num_sample_points

        loss_reg = self.loss_reg(
            pred_sample_points,     # (B*N_q, N_sample_points=100, 2)
            gt_sample_points,       # (B*N_q, N_sample_points=100, 2),
            weight=reg_weight,      # (B*N_q, N_sample_points=100, 2)
            avg_factor=avg_factor   # N_pos * N_sample_points
        )
        loss_dict['loss_reg'] = loss_reg

        if self.with_seg:
            pred_seg = preds_dicts['pred_seg']      # (B, num_class=1, img_H, img_W)
            gt_semantic_seg = gt_semantic_seg.squeeze(dim=1).long()      # (B, img_H, img_W)
            pred_seg = torch.nn.functional.interpolate(pred_seg, size=gt_semantic_seg.shape[-2:],
                                                       mode='bilinear', align_corners=True)
            pred_seg = pred_seg.squeeze(dim=1)      # (B, img_H, img_W)

            loss_seg = self.loss_seg(
                pred_seg,
                gt_semantic_seg,
                1 / self.seg_pos_weight,   # (B*num_priors, )
            )

            loss_dict['loss_seg'] = loss_seg

        return loss_dict

    def get_targets(self, cls_logits_list, pred_control_points_list, gt_control_points, gt_labels_list):
        """
        :param cls_logits_list:     # List[(N_q, 1), (N_q, 1), ...]    len = batch_size
        :param pred_control_points_list:  # List[(N_q, 4, 2), (N_q, 4, 2), ...]  len = batch_size
        :param gt_control_points:  #  List[(N_gt0, 4, 2), (N_gt1, 4, 2), ...]     len = batch_size
        :param gt_labels:  List[(N_gt0,), (N_gt1,), ...]
        :return:
            labels_list: List[(N_q, ), (N_q, ), ...]
            label_weights_list: List[(N_q, ), (N_q, ), ...]
            lane_targets_list: List[(N_q, 4, 2), (N_q, 4, 2), ...]
            lane_weights_list: List[(N_q, 4, 2), (N_q, 4, 2), ...]
        """
        (labels_list, label_weights_list, control_points_targets_list, control_points_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_logits_list, pred_control_points_list, gt_control_points, gt_labels_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, control_points_targets_list, control_points_weights_list,
                num_total_pos, num_total_neg)

    def _get_target_single(self,
                           cls_logits,
                           pred_control_points,
                           gt_control_points,
                           gt_labels):
        """"
        Args:
            cls_logits (Tensor): (N_q, n_cls=1)
            pred_control_points (Tensor): (N_q, 4, 2)
            gt_control_points (Tensor): (N_gt, 4, 2)
            gt_labels (Tensor): (N_gt, )
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.   (N_q, )
                - label_weights (Tensor]): Label weights of each image.     (N_q, )
                - control_points_targets (Tensor): Lane targets of each image.  (N_q, 4, 2)
                - control_points_weights (Tensor): Lane weights of each image.  (N_q, 4, 2)
                - pos_inds (Tensor): Sampled positive indices for each image.   (N_pos, )
                - neg_inds (Tensor): Sampled negative indices for each image.   (N_neg, )
        """
        num_preds = pred_control_points.size(0)
        with torch.no_grad():
            # (N_q, ),  (N_q, )
            assigned_gt_inds, assigned_labels = self.assigner.assign(cls_logits, pred_control_points,
                                                                     gt_control_points, gt_labels)

        # sampler
        pos_inds = torch.nonzero(
            assigned_gt_inds > 0, as_tuple=False).squeeze(-1).unique()     # (N_pos, )
        neg_inds = torch.nonzero(
            assigned_gt_inds == 0, as_tuple=False).squeeze(-1).unique()    # (N_neg, )
        pos_assigned_gt_inds = assigned_gt_inds[pos_inds] - 1       # (N_pos, ), pos样本对应的gt索引(0-based)
        pos_gt_labels = assigned_labels[pos_inds]       # (N_pos, )

        if len(gt_control_points) == 0:
            # hack for index error case
            assert pos_assigned_gt_inds.numel() == 0
            pos_gt_control_points = torch.empty_like(gt_control_points)  # (N_gt, 4, 2)
        else:
            pos_gt_control_points = gt_control_points[pos_assigned_gt_inds.long(), :]   # (N_pos, 4, 2)

        # label targets
        # 默认为bg
        labels = cls_logits.new_full((num_preds,), fill_value=self.num_classes, dtype=torch.long)   # (N_q, )
        assert (pos_gt_labels == 0).all(), "pos_gt_labels 有问题"
        label_weights = cls_logits.new_zeros((num_preds,), dtype=torch.float)   # (N_q, )

        # lane targets
        control_points_targets = torch.zeros_like(pred_control_points)    # (N_q, 4, 2)
        control_points_weights = pred_control_points.new_zeros((num_preds, ))   # (N_q, )

        if len(pos_inds) > 0:
            labels[pos_inds] = pos_gt_labels
            label_weights[pos_inds] = 1.0
            control_points_targets[pos_inds] = pos_gt_control_points
            control_points_weights[pos_inds] = 1.0

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        assert label_weights.all(), "label_weights 有问题, 分类时应当考虑所有样本."
        return (labels, label_weights, control_points_targets, control_points_weights,
                pos_inds, neg_inds)

    @force_fp32(apply_to=('preds_dicts'))
    def get_lanes(self, preds_dicts, img_metas, cfg=None):
        cfg = self.test_cfg if cfg is None else cfg
        exist_thresh = cfg['score_thr']
        window_size = cfg['window_size']
        max_lanes = cfg['max_lanes']

        output_cls_logits = preds_dicts['logits'].squeeze(dim=-1)     # (B, N_q)
        output_control_points = preds_dicts['pred_control_points']    # (B, N_q, 4, 2)
        num_pred = output_cls_logits.shape[1]

        output_existence_confs = output_cls_logits.sigmoid()     # (B, N_q)
        out_existences = output_existence_confs > exist_thresh   # (B, N_q)

        if window_size > 0:
            _, max_indices = F.max_pool1d(output_existence_confs.unsqueeze(dim=1).contiguous(),
                                          kernel_size=window_size,
                                          stride=1,
                                          padding=(window_size - 1) // 2,
                                          return_indices=True)  # (B, 1, N_q)
            max_indices = max_indices.squeeze(1)    # (B, N_q)
            indices = torch.arange(0, num_pred, dtype=output_existence_confs.dtype,
                                   device=output_existence_confs.device).unsqueeze(dim=0).expand_as(max_indices)
            local_maxima = (max_indices == indices)   # (B, N_q)
            out_existences *= local_maxima   # (B, N_q)

        results_list = []
        for existence_conf, pred_control_points, existence, img_meta in zip(
                output_existence_confs, output_control_points, out_existences, img_metas):
            # existence_conf: (N_q, )
            # pred_control_points: (N_q, 4, 2)
            # existence: (N_q, )

            valid_scores = existence_conf * existence
            sorted_scores, sorted_indices = torch.sort(valid_scores, dim=0, descending=True)
            valid_indices = torch.nonzero(sorted_scores, as_tuple=True)[0][:max_lanes]

            keep_index = sorted_indices[valid_indices]      # (N_keep, )
            scores = existence_conf[keep_index]             # (N_keep, )
            pred_control_points = pred_control_points[keep_index]       # (N_keep, 4, 2)

            if len(keep_index) == 0:
                results_list.append([])
                continue
            # List[(N0, 2),  (N1, 2), ...]
            pred = self.predictions_to_pred(scores, pred_control_points, img_meta)
            results_list.append(pred)

        return results_list

    def predictions_to_pred(self, scores, pred_control_points, img_meta):
        """
        :param scores: (N_keep, )
        :param pred_control_points: (N_keep, 4, 2)
        :param img_meta:
        :return:
            lanes: List[(N0, 2),  (N1, 2), ...]
        """
        ori_height, ori_weight = img_meta['ori_shape'][:2]
        crop = img_meta.get('crop', (0, 0, 0, 0))     # (x_min, y_min, x_max, y_max)
        y_min = crop[1]
        dataset = self.test_cfg.get('dataset', None)

        lanes = []
        for cur_score, cur_pred_control_points in zip(scores, pred_control_points):
            # cur_score: (1, )
            # cur_pred_control_points: (4, 2)
            cur_score = cur_score.detach().cpu().numpy()
            cur_pred_control_points = cur_pred_control_points.detach().cpu().numpy()

            if dataset == 'tusimple':
                ppl = 56
                gap = 10
                bezier_threshold = 5.0 / ori_height
                h_samples = np.array([1.0 - (ppl - i) * gap / ori_height for i in range(ppl)], dtype=np.float32)   # (56, )

                cur_pred_sample_points = self.bezier_curve.get_sample_points(
                    control_points_matrix=cur_pred_control_points,
                    num_sample_points=ori_height)  # (N_sample_points-720, 2)  2: (x, y)

                ys = (cur_pred_sample_points[:, 1] * (ori_height - y_min) + y_min) / ori_height   # (720, )
                dis = np.abs(h_samples.reshape(ppl, -1) - ys)    # (56, 720)
                idx = np.argmin(dis, axis=-1)  # (56, )
                temp = []
                for i in range(ppl):
                    h = ori_height - (ppl - i) * gap
                    if dis[i][idx[i]] > bezier_threshold or cur_pred_sample_points[idx[i]][0] > 1 \
                            or cur_pred_sample_points[idx[i]][0] < 0:
                        temp.append([-2, h])
                    else:
                        temp.append([cur_pred_sample_points[idx[i]][0] * ori_weight, h])
                temp = np.array(temp, dtype=np.float32)
                lanes.append(temp)
            else:
                cur_pred_sample_points = self.bezier_curve.get_sample_points(
                    control_points_matrix=cur_pred_control_points,
                    num_sample_points=self.test_cfg['num_sample_points'])      # (N_sample_points, 2)  2: (x, y)

                lane_xs = cur_pred_sample_points[:, 0]      # 由上向下
                lane_ys = cur_pred_sample_points[:, 1]

                x_mask = np.logical_and(lane_xs >= 0, lane_xs < 1)
                y_mask = np.logical_and(lane_ys >= 0, lane_ys < 1)
                mask = np.logical_and(x_mask, y_mask)

                lane_xs = lane_xs[mask]
                lane_ys = lane_ys[mask]
                lane_ys = (lane_ys * (ori_height - y_min) + y_min) / ori_height
                if len(lane_xs) <= 1:
                    continue
                points = np.stack((lane_xs, lane_ys), axis=1)  # (N_sample_points, 2)  normalized

                points = sorted(points, key=lambda x: x[1])
                filtered_points = []
                used = set()
                for p in points:
                    if p[1] not in used:
                        filtered_points.append(p)
                        used.add(p[1])
                points = np.array(filtered_points)

                lane = Lane(points=points,
                            metadata={
                                'conf': cur_score,
                            })
                lanes.append(lane)
        return lanes






















