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
from mmcv.cnn import bias_init_with_prob


@HEADS.register_module()
class LaneATTHead(BaseDenseHead):
    def __init__(self,
                 in_channels=512,
                 num_points=72,
                 img_size=(640, 360),
                 stride=32,
                 anchor_feat_channels=64,
                 anchors_freq_path=None,
                 topk_anchors=None,
                 return_attention_matrix=True,
                 num_classes=1,
                 loss_cls=dict(
                     type='SoftmaxFocalloss',
                     use_sigmoid=False,
                     gamma=2.0,
                     alpha=0.25,
                     reduction='mean',
                     loss_weight=1.0
                 ),
                 loss_reg=dict(
                     type='SmoothL1Loss',
                     beta=1.0,
                     reduction='mean',
                     loss_weight=1.0
                 ),
                 train_cfg=None,
                 test_cfg=None,
                 sync_cls_avg_factor=False,
                 init_cfg=None,
                 ):
        super(LaneATTHead, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.n_strips = num_points - 1
        self.n_offsets = num_points
        self.img_w, self.img_h = img_size
        self.fmap_w, self.fmap_h = self.img_w // stride,  self.img_h // stride
        self.anchor_ys = torch.linspace(1, 0, steps=self.n_offsets, dtype=torch.float32)
        self.anchor_cut_ys = torch.linspace(1, 0, steps=self.fmap_h, dtype=torch.float32)
        self.anchor_feat_channels = anchor_feat_channels
        self.stride = stride
        self.num_classes = num_classes
        self.return_attention_matrix = return_attention_matrix

        # Anchor angles, same ones used in Line-CNN
        self.left_angles = [72., 60., 49., 39., 30., 22.]
        self.right_angles = [108., 120., 131., 141., 150., 158.]
        self.bottom_angles = [165., 150., 141., 131., 120., 108., 100., 90., 80., 72., 60., 49., 39., 30., 15.]
        # Generate anchors
        self.anchors, self.anchors_cut = self.generate_anchors(lateral_n=72, bottom_n=128)
        # self.anchors: (n_anchors, 3 + S), where n_anchors = N_origins*N_angles
        # self.anchors_cut: (n_anchors, 3 + H_f)   3: start_y, start_x, len

        # Filter masks if `anchors_freq_path` is provided
        if anchors_freq_path is not None:
            anchors_mask = torch.load(anchors_freq_path).cpu()      # (n_anchors, )
            assert topk_anchors is not None
            # 根据被匹配为正样本的次数进行排序, 并只保留前topk_anchors个anchors  for speed efficiency.
            ind = torch.argsort(anchors_mask, descending=True)[:topk_anchors]
            self.anchors = self.anchors[ind]    # (topk_anchors, 3 + S)
            self.anchors_cut = self.anchors_cut[ind]    # (topk_anchors, 3 + H_f)

        # anchor_coord: (n_anchors， H_f, 2)   2: (v, u)
        # invalid_mask: (n_anchors, H_f)
        self.anchor_coord, self.invalid_mask = self.compute_anchor_cut_indices(
            self.fmap_w, self.fmap_h)

        # Setup and initialize layers
        self.conv1 = nn.Conv2d(in_channels, self.anchor_feat_channels, kernel_size=1)
        self.cls_layer = nn.Linear(2 * self.anchor_feat_channels * self.fmap_h, 2)
        self.reg_layer = nn.Linear(2 * self.anchor_feat_channels * self.fmap_h, self.n_offsets + 1)
        self.attention_layer = nn.Linear(self.anchor_feat_channels * self.fmap_h, len(self.anchors) - 1)
        self.initialize_layer(self.attention_layer)
        self.initialize_layer(self.conv1)
        self.initialize_layer(self.cls_layer)
        self.initialize_layer(self.reg_layer)

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor

        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is LaneATTHead):
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[0] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        self.loss_cls = build_loss(loss_cls)
        self.loss_reg = build_loss(loss_reg)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.anchors = self.anchors.cuda()
        self.anchor_ys = self.anchor_ys.cuda()
        self.anchors_cut = self.anchors_cut.cuda()
        self.invalid_mask = self.invalid_mask.cuda()

    @staticmethod
    def initialize_layer(layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            torch.nn.init.normal_(layer.weight, mean=0., std=0.001)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)

    def generate_anchors(self, lateral_n, bottom_n):
        """
        Args:
            lateral_n: 72
            bottom_n: 128
        Returns:
            anchors: List
            cut_anchors: List
        """
        left_anchors, left_cut = self.generate_side_anchors(self.left_angles, x=0., nb_origins=lateral_n)
        right_anchors, right_cut = self.generate_side_anchors(self.right_angles, x=1., nb_origins=lateral_n)
        bottom_anchors, bottom_cut = self.generate_side_anchors(self.bottom_angles, y=1., nb_origins=bottom_n)

        return torch.cat([left_anchors, bottom_anchors, right_anchors]), torch.cat([left_cut, bottom_cut, right_cut])

    def generate_side_anchors(self, angles, nb_origins, x=None, y=None):
        """
        Args:
            angles: Anchor angles
            nb_origins: 该边界处产生的车道线Anchors数量.
        Returns:
            anchors: (n_anchors, 3+S)
            anchors_cut: (n_anchors, 3+H_f)
        """
        if x is None and y is not None:
            starts = [(x, y) for x in np.linspace(1., 0., num=nb_origins)]
        elif x is not None and y is None:
            starts = [(x, y) for y in np.linspace(1., 0., num=nb_origins)]
        else:
            raise Exception('Please define exactly one of `x` or `y` (not neither nor both)')

        n_anchors = nb_origins * len(angles)   # 该边界处产生的Anchors总数.

        # each row, first for x and second for y:
        # 1 start_y, start_x, 1 lenght, S coordinates
        anchors = torch.zeros((n_anchors, 2 + 1 + self.n_offsets))
        anchors_cut = torch.zeros((n_anchors, 2 + 1 + self.fmap_h))
        for i, start in enumerate(starts):
            for j, angle in enumerate(angles):
                k = i * len(angles) + j
                anchors[k] = self.generate_anchor(start, angle)
                anchors_cut[k] = self.generate_anchor(start, angle, cut=True)

        return anchors, anchors_cut

    def generate_anchor(self, start, angle, cut=False):
        """
        Args:
            start: (start_x, start_y)
            angle: float
        Returns:
            anchor: cut=False --> (2+1+S, )  /  cut=True --> (2+1+H_f, )
        """
        if cut:
            anchor_ys = self.anchor_cut_ys  # len = H_f: [1, ..., 0]
            anchor = torch.zeros(2 + 1 + self.fmap_h)
        else:
            anchor_ys = self.anchor_ys  # len = S: [1, ..., 0]
            anchor = torch.zeros(2 + 1 + self.n_offsets)

        angle = angle * math.pi / 180.  # degrees to radians
        start_x, start_y = start
        anchor[0] = 1 - start_y     # anchor[2]中保留的是lane的start_y, 0表示图像底部，1表示图像顶部.
        anchor[1] = start_x     # normalized
        # 与anchor_ys对应的x坐标, anchor_ys从图像底部-->顶部（1-->0）.
        anchor[3:] = (start_x + (1 - anchor_ys - 1 + start_y) / math.tan(angle)) * self.img_w

        return anchor

    def compute_anchor_cut_indices(self, fmaps_w, fmaps_h):
        """
        Args:
            fmaps_w: W_f
            fmaps_h: H_f
        Returns:
            anchor_coord: (n_anchors， H_f, 2)   2: (v, u)
            invalid_mask: (n_anchors, H_f)
        """
        # definitions
        n_proposals = len(self.anchors_cut)     # n_anchors = N_origins*N_angles

        # indexing
        # 与anchor_ys对应的x坐标,  anchor_ys(len=H_f)表示从图像底部-->顶部（1-->0）
        # flip后, 表示从图像顶部-->底部:  (n_anchors， H_f)
        unclamped_xs = torch.flip((self.anchors_cut[:, 3:] / self.stride).round().long(), dims=(1,))
        cut_xs = torch.clamp(unclamped_xs, 0, fmaps_w - 1)   # (n_anchors, H_f), 限定x坐标范围
        invalid_mask = (unclamped_xs < 0) | (unclamped_xs > fmaps_w)    # 超出图像范围的mask (n_anchors， H_f)

        cut_ys = torch.arange(0, fmaps_h)   # (H_f, )
        cut_ys = cut_ys[None, :].repeat(n_proposals, 1)     # (n_anchors， H_f)
        anchor_coord = torch.stack([cut_ys, cut_xs], dim=-1)    # (n_anchors， H_f, 2)   2: (v, u)

        return anchor_coord, invalid_mask

    def draw_anchors(self, k=None):
        base_ys = self.anchor_ys.cpu().numpy()    # (S, )  (1, ..., 0)
        img = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)   # (img_h, img_w, 3)
        i = -1
        for anchor in self.anchors:
            i += 1
            if k is not None and i != k:
                continue
            anchor = anchor.cpu().numpy()
            xs = anchor[3:]
            ys = base_ys * self.img_h
            points = np.vstack((xs, ys)).T.round().astype(int)
            for p_curr, p_next in zip(points[:-1], points[1:]):
                img = cv2.line(img, tuple(p_curr), tuple(p_next), color=(255, 0, 0), thickness=1)

        return img

    def forward(self, x):
        """
        :param x: (B, C_in, fH, fW)
        :return:
            output = {
                'cls_logits': (B, n_anchors, 2)
                'lanes_preds': (B, n_anchors, 3+S)   1 start_y, 1 start_x, 1 length, S coordinates
                'attention_matrix': (B, n_anchors, n_anchors)
            }
        """
        x = x[-1]
        batch_features = self.conv1(x)      # (B, C_in, fH, fW) --> (B, C, fH, fW)
        batch_anchor_features = self.cut_anchor_features(batch_features)    # (B, n_anchors, C, fH)

        # Join proposals from all images into a single proposals features batch
        # (B, n_anchors, C, fH) --> (B*n_anchors, C*fH)
        batch_anchor_features = batch_anchor_features.view(-1, self.anchor_feat_channels * self.fmap_h)

        # Add attention features
        softmax = nn.Softmax(dim=1)
        # (B*n_anchors, C*H) --> (B*n_anchors, n_anchors-1)
        scores = self.attention_layer(batch_anchor_features)
        # (B*n_anchors, n_anchors-1)  --> (B, n_anchors, n_anchors-1)
        attention = softmax(scores).reshape(x.shape[0], len(self.anchors), -1)
        # (n_anchors, n_anchors) --> (B, n_anchors, n_anchors)
        attention_matrix = torch.eye(attention.shape[1], device=x.device).repeat(x.shape[0], 1, 1)
        non_diag_inds = torch.nonzero(attention_matrix == 0., as_tuple=False)
        attention_matrix[:] = 0
        attention_matrix[non_diag_inds[:, 0], non_diag_inds[:, 1], non_diag_inds[:, 2]] = attention.flatten()

        # (B*n_anchors, C * H) --> (B, n_anchors, C*H)
        batch_anchor_features = batch_anchor_features.reshape(x.shape[0], len(self.anchors), -1)
        # (B, n_anchors, n_anchors) @ (B, n_anchors, C*fH) --> (B, n_anchors, C*fH)
        attention_features = torch.bmm(attention_matrix, batch_anchor_features)
        # (B, n_anchors, C*fH) --> (B*n_anchors, C*fH)    paper中的global feature
        attention_features = attention_features.reshape(-1, self.anchor_feat_channels * self.fmap_h)
        # (B, n_anchors, C*H) --> (B*n_anchors, C*H)    paper中的local feature
        batch_anchor_features = batch_anchor_features.reshape(-1, self.anchor_feat_channels * self.fmap_h)
        # (B*n_anchors, 2*C*H)
        batch_anchor_features = torch.cat((attention_features, batch_anchor_features), dim=1)

        # Predict
        # (B*n_anchors, 2*C*H) --> (B*n_anchors, 2)
        cls_logits = self.cls_layer(batch_anchor_features)
        # (B*n_anchors, 2*C*H) --> (B*n_anchors,  S(n_offsets)+1(l))
        reg = self.reg_layer(batch_anchor_features)

        # Undo joining
        # (B*n_anchors, 2) --> (B, n_anchors, 2)
        cls_logits = cls_logits.reshape(x.shape[0], -1, cls_logits.shape[1])
        # (B*n_anchors, 1(l) + S(n_offsets))  --> (B, n_anchors, 1+S)
        reg = reg.reshape(x.shape[0], -1, reg.shape[1])

        # Add offsets to anchors
        # (B, n_anchors, 3+S)  1 start_y, 1 start_x, 1 length, S coordinates
        lanes_preds = self.anchors.repeat(x.shape[0], 1, 1).clone()
        lanes_preds[:, :, 2] = reg[:, :, 0]      # l
        lanes_preds[:, :, 3:] += reg[:, :, 1:]    # x + x_offsets

        output = {
            'cls_logits': cls_logits,   # (B, n_anchors, 2)
            'lanes_preds': lanes_preds,     # (B, n_anchors, 3+S)
        }
        if self.return_attention_matrix:
            output['attention_matrix'] = attention_matrix

        return output

    def cut_anchor_features(self, features):
        """
        Args:
            features: (B, C=64, fH, fW)
        Returns:
            batch_anchor_features: (B, n_anchors, C, fH)
        """
        batch_size, C = features.shape[:2]

        # (B, C, n_anchors, fH)
        batch_anchor_features = features[:, :, self.anchor_coord[:, :, 0], self.anchor_coord[:, :, 1]]
        # (n_anchors, fH) --> (B, C, n_anchors, fH)
        invalid_mask = self.invalid_mask[None, None, ...].repeat(batch_size, C, 1, 1)
        batch_anchor_features[invalid_mask] = 0

        # (B, C, n_anchors, fH) --> (B, n_anchors, C, fH)
        batch_anchor_features = batch_anchor_features.permute(0, 2, 1, 3).contiguous()
        return batch_anchor_features

    def nms(self, batch_cls_logits, batch_lanes_preds, cfg):
        """   只用于train阶段，根据nms_thresh进行NMS.
        :param batch_cls_logits: (B, n_anchors, 2)
        :param batch_lanes_preds: (B, n_anchors, 3+S)
        :param cfg:
        :return:
            proposals_list
        """
        nms_pre = cfg.get('nms_pre', len(self.anchors))
        proposals_list = []
        softmax = nn.Softmax(dim=1)

        for cls_logits, lane_preds in zip(batch_cls_logits, batch_lanes_preds):
            # cls_logits: (n_anchors, 2)
            # lane_preds: (n_anchors, 3+S)

            with torch.no_grad():
                if self.use_sigmoid_cls:
                    scores = cls_logits.sigmoid()
                else:
                    scores = softmax(cls_logits)[:, 0]    # (n_anchors, )  fg score

                keep, num_to_keep, _ = nms(
                    lane_preds.clone(),    # (N, 3+S)
                    scores,             # (N, )
                    overlap=cfg.nms_thres,
                    top_k=cfg.max_lanes,
                )
                keep = keep[:num_to_keep]       # (N_keep, )

            cls_logits = cls_logits[keep]   # (N_keep, )
            lane_preds = lane_preds[keep]   # (N_keep, 3+S)
            anchors = self.anchors[keep]    # (N_keep, 3+S)

            proposals_list.append((cls_logits, lane_preds, anchors))

        return proposals_list

    def loss(self,
             preds_dicts,
             img_metas,
             gt_lanes,
             gt_labels,
             gt_semantic_seg=None,
             ):
        """Compute losses of the head.
        Args:
            preds_dicts: dict{
                'cls_logits': cls_logits,   # (B, n_anchors, 2)
                'lanes_preds': lanes_preds,     # (B, n_anchors, 3+S)
                'attention_matrix': (B, n_anchors, n_anchors)
            }
            gt_lanes (List):  List[(N_lanes0, 4+S), (N_lanes1, 4+S), ...]
                3+S: 1 start_y (normalized), 1 start_x (absolute), 1 length (absolute),
                     S coordinates(absolute)
            gt_labels (List): List[(N_lanes0, ), (N_lanes1, ), ...]
            gt_semantic_seg (Tensor): (B, 1, H, W)
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        cls_logits = preds_dicts['cls_logits']
        lanes_preds = preds_dicts['lanes_preds']

        batch_size = cls_logits.shape[0]

        batch_proposals_list = self.nms(cls_logits, lanes_preds, self.train_cfg)
        # List: [Tuple((N, 2), (N, 3+S), (N, )), Tuple((N, 2), (N, 3+S), (N, )), ...]

        losses_cls, losses_reg = multi_apply(
            self.loss_single, batch_proposals_list, gt_lanes, gt_labels
        )

        loss_cls = sum(losses_cls) / batch_size
        loss_reg = sum(losses_reg) / batch_size

        loss_dict = dict()
        loss_dict['loss_cls'] = loss_cls
        loss_dict['loss_reg'] = loss_reg

        return loss_dict

    def loss_single(self,
                    proposals,
                    gt_lanes,
                    gt_labels
                    ):
        """
        :param proposals: Tuple(
            'cls_logits': (N_pred, 2)
            'lane_preds': (N_pred, 3+S)
            'anchor_inds': (N_pred, )
        )
        :param gt_lanes: (N_gt, 3+S)
            3+S: 1 start_y (normalized), 1 start_x (absolute), 1 length (absolute),
                 S coordinates(absolute)
        :param gt_labels: (N_gt, )
        :return:
        """
        # cls_logits: (N_pred, 2)
        # lane_preds: (N_pred, 3+S)
        # anchors: (N_pred, 3+S)
        cls_logits, lane_preds, anchors = proposals

        cls_reg_targets = self.get_targets(
            cls_logits,
            lane_preds,
            anchors,
            gt_lanes,
            gt_labels
        )
        # cls_reg_targets:
        # labels: (N_pred,)
        # label_weights: (N_pred,)
        # lane_targets: (N_pred, 3 + S)
        # lane_weights: (N_pred, 3 + S)
        # pos_inds: (N_pos, )
        # neg_inds: (N_neg, )

        labels, label_weights, lane_targets, lane_weights, pos_inds, neg_inds = cls_reg_targets
        num_total_pos = len(pos_inds)
        num_total_neg = len(neg_inds)

        # 1. classification loss
        cls_scores = cls_logits.view(-1, cls_logits.shape[-1])      # (N_pred, n_cls)
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1.0)
        loss_cls = self.loss_cls(
            cls_scores,      # (N_pred, n_cls)
            labels,          # (N_pred, )
            label_weights,   # (N_pred, )
            avg_factor=cls_avg_factor)

        # 2. reg loss
        pos_preds = lane_preds[pos_inds]    # (N_pos, 3+S)  3: start_y, start_x, len, S
        pos_targets = lane_targets[pos_inds]        # (N_pos, 3+S)  3: start_y, start_x, len, S

        reg_pred = pos_preds[:, 2:]  # (N_pos, 1+S)
        with torch.no_grad():
            positive_starts = (pos_preds[:, 0] * self.n_strips).round().long()   # (N_pos, ) proposal start_y
            target_starts = (pos_targets[:, 0] * self.n_strips).round().long()   # (N_pos, ) target start_y
            pos_targets[:, 2] -= positive_starts - target_starts
            ends = (positive_starts + pos_targets[:, 2] - 1).round().long()  # 其实就是end_gt

            # (N_pos, 1+S+1)    # 1+S+1: length + S + pad
            invalid_offsets_mask = lane_preds.new_zeros((num_total_pos, 1 + self.n_offsets + 1),
                                                         dtype=torch.int)
            all_indices = torch.arange(num_total_pos, dtype=torch.long)
            invalid_offsets_mask[all_indices, 1 + positive_starts] = 1
            invalid_offsets_mask[all_indices, 1 + ends + 1] -= 1
            invalid_offsets_mask = invalid_offsets_mask.cumsum(dim=1) == 0
            invalid_offsets_mask = invalid_offsets_mask[:, :-1]
            invalid_offsets_mask[:, 0] = False

            reg_target = pos_targets[:, 2:]  # (N_pos, 1+S)
            reg_target[invalid_offsets_mask] = reg_pred[invalid_offsets_mask]

        loss_reg = self.loss_reg(
            reg_pred,
            reg_target,
        )

        return loss_cls, loss_reg

    def assign(self, anchors, gt_lanes, gt_labels, t_pos=15., t_neg=20.):
        """
        :param anchors: (N_pred, 3+S)     3: start_y(normalized), start_x(normalized), len(absolute), S(absolute)
        :param gt_lanes: (N_gt, 3+S)      3: start_y(normalized), start_x(absolute), len(absolute), S(absolute)
        :param gt_labels: (N_gt, )
        :return:
            assigned_gt_inds: (N_pred, )  pos: gt_id+1,  neg: 0,  ignore: -1
            assigned_labels: (N_pred, )   pos: gt_label  other: -1
        """
        num_proposals = anchors.shape[0]
        num_targets = gt_lanes.shape[0]
        # 1. assign -1 by default
        assigned_gt_inds = anchors.new_full((num_proposals,), -1, dtype=torch.long)
        assigned_labels = anchors.new_full((num_proposals,), -1, dtype=torch.long)

        if num_targets == 0 or num_proposals == 0:
            # No ground truth or boxes, return empty assignment
            if num_targets == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return assigned_gt_inds, assigned_labels

        # pad proposals and target for the valid_offset_mask's trick
        proposals_pad = anchors.new_zeros(num_proposals, anchors.shape[1] + 1)  # (N_proposal, 3+S+1)
        proposals_pad[:, :-1] = anchors  # (N_proposal, 3+S)
        targets_pad = gt_lanes.new_zeros(num_targets, gt_lanes.shape[1] + 1)  # (N_lanes, 3+S+1)
        targets_pad[:, :-1] = gt_lanes  # (N_lanes, 3+S)

        # (N_proposal, N_gt, 3+S)
        proposals = proposals_pad.unsqueeze(dim=1).repeat(1, num_targets, 1)
        # (N_proposal, N_gt, 3+S)
        targets = targets_pad.unsqueeze(dim=0).repeat(num_proposals, 1, 1)

        proposal_coords = proposals[..., 3:]   # (N_proposal, N_gt, S)  absolute
        targets_coords = targets[..., 3:]      # (N_proposal, N_gt, S)  absolute

        # get start and the intersection of offsets
        target_starts = targets[..., 0] * self.n_strips      # (N_proposal, N_gt)
        proposals_starts = proposals[..., 0] * self.n_strips    # (N_proposal, N_gt)
        starts = torch.max(target_starts, proposals_starts).round().long()  # start_y   (N_proposal, N_gt)
        ends = (target_starts + targets[..., 2] - 1).round().long()         # end_y = e_gt    (N_proposal, N_gt)
        lengths = ends - starts + 1     # (N_proposal, N_gt)
        ends[lengths < 0] = starts[lengths < 0] - 1
        lengths[lengths < 0] = 0    # a negative number here means no intersection, thus zero length

        valid_offsets_mask = targets.new_zeros(targets_coords.shape)       # (N_proposal, N_gt, S+1)
        valid_offsets_mask.scatter_(2, starts.unsqueeze(dim=-1).long(), 1.0)
        # 对于length<=0, 会将start处的1减掉, valid_offsets_mask全为0.
        valid_offsets_mask.scatter_(2, (ends + 1).unsqueeze(dim=-1).long(), -1.0, reduce='add')
        valid_offsets_mask = valid_offsets_mask.cumsum(dim=-1) != 0.  # (N_proposal, N_gt, S+1)

        # compute distances
        # (N_proposal, N_lanes, S+1) --> (N_proposal, N_lanes)
        distances = torch.abs((targets_coords - proposal_coords) * valid_offsets_mask.float()).sum(dim=-1) / (
                    lengths.float() + 1e-9
                    )  # avoid division by zero

        INFINITY = 987654.
        distances[lengths == 0] = INFINITY
        distances = distances.view(num_proposals, num_targets)  # (N_proposal, N_gt)

        # 距离小于t_pos时为正样本, 大于t_neg时为负样本, 忽略其余的样本.
        positives = distances.min(dim=1)[0] < t_pos  # (N_proposal, )
        negatives = distances.min(dim=1)[0] > t_neg  # (N_proposal, )

        pos_gt_inds = distances[positives].argmin(dim=-1).long()   # (N_pos, )
        assigned_gt_inds[positives] = pos_gt_inds + 1
        assigned_labels[positives] = gt_labels[pos_gt_inds]

        assigned_gt_inds[negatives] = 0
        return assigned_gt_inds, assigned_labels

    def get_targets(self, cls_logits, lane_preds, anchors, gt_lanes, gt_labels):
        """
        :param cls_logits: (N_pred, 2)
        :param lane_preds: (N_pred, 3+S)
        :param anchors: (N_pred, 3+S)
        :param gt_lanes: (N_gt, 3+S)
        :param gt_labels: (N_gt, )
        :return:
            labels: (N_pred, )
            label_weights: (N_pred, )
            lane_targets: (N_pred, 3+S)
            lane_weights: (N_pred, 3+S)
            pos_inds: (N_pos, )
            neg_inds: (N_neg, )
        """
        num_proposals = cls_logits.shape[0]
        with torch.no_grad():
            # 利用anchor（而不是proposal）和gt进行匹配!!! 使用proposal进行assign会造成训练不稳定.
            # (N_pred, ),  (N_pred, )
            assigned_gt_inds, assigned_labels = self.assign(anchors, gt_lanes, gt_labels,
                                                            t_pos=self.train_cfg.get('t_pos', 15),
                                                            t_neg=self.train_cfg.get('t_neg', 20)
                                                            )

        # sampler
        pos_inds = torch.nonzero(
            assigned_gt_inds > 0, as_tuple=False).squeeze(-1).unique()     # (N_pos, )
        neg_inds = torch.nonzero(
            assigned_gt_inds == 0, as_tuple=False).squeeze(-1).unique()    # (N_neg, )

        pos_assigned_gt_inds = assigned_gt_inds[pos_inds] - 1       # (N_pos, ) pos样本对应的gt索引(0-based)
        pos_gt_labels = assigned_labels[pos_inds]       # (N_pos, )

        if len(gt_lanes) == 0:
            # hack for index error case
            assert pos_assigned_gt_inds.numel() == 0
            pos_gt_lanes = torch.empty_like(gt_lanes)
        else:
            pos_gt_lanes = gt_lanes[pos_assigned_gt_inds.long(), :]   # (N_pos, 3+S)

        # label targets
        # 默认为bg
        labels = cls_logits.new_full((num_proposals, ), fill_value=self.num_classes, dtype=torch.long)
        label_weights = cls_logits.new_zeros((num_proposals, ), dtype=torch.float)
        # lane targets
        # (num_priors, 3+S)
        lane_targets = torch.zeros_like(lane_preds)
        lane_weights = torch.zeros_like(lane_preds)

        if len(pos_inds) > 0:
            labels[pos_inds] = pos_gt_labels
            label_weights[pos_inds] = 1.0
            lane_targets[pos_inds] = pos_gt_lanes  # (N_pos, 3+S)
            lane_weights[pos_inds] = 1.0

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        return (labels, label_weights, lane_targets, lane_weights,
                pos_inds, neg_inds)

    @force_fp32(apply_to=('preds_dicts'))
    def get_lanes(self, preds_dicts, img_metas, cfg=None):
        '''
        Convert model output to lanes.
        Args:
            preds_dicts = dict {
                'cls_logits': (B, N_pred, 2)
                'lane_preds': (B, N_pred, 3+S)
            }
        Returns:
            results_list: Lane results of each image. The outer list corresponds to each image. The inner list
                corresponds to each lane.
                List[List[lane0, lane1, ...], ...]
        '''
        output_cls_logits = preds_dicts['cls_logits']   # (B, N_pred, 2)
        output_lane_preds = preds_dicts['lanes_preds']   # (B, N_pred, 3+S)

        cfg = self.test_cfg if cfg is None else cfg
        nms_pre = cfg.get('nms_pre', len(self.anchors))

        results_list = []
        for cls_logits, lanes_preds, img_meta in zip(output_cls_logits, output_lane_preds, img_metas):
            # cls_scores: (N_pred, n_cls=2)
            # lanes_preds: (N_pred, 3+S)
            if self.use_sigmoid_cls:
                scores = cls_logits.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_logits.softmax(-1)[:, :-1]

            # filter out the conf lower than conf threshold
            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre, dict(lanes_preds=lanes_preds)
            )
            # scores: (N, )    # N = nms_pre if num_all_valid_anchors > nms_pre else num_all_valid_anchors
            # labels: (N, )
            # keep_idxs: (N, )
            # filtered_results: {'lanes_preds': (N, 3+S)}
            scores, labels, keep_idxs, filtered_results = results
            lanes_preds = filtered_results['lanes_preds']   # (N, 3+S)

            if lanes_preds.shape[0] == 0:
                results_list.append([])
                continue

            scores = scores.detach().clone()
            lanes_preds = lanes_preds.detach().clone()
            keep, num_to_keep, _ = nms(
                lanes_preds,    # (N, 3+S)
                scores,             # (N, )
                overlap=cfg.nms_thres,
                top_k=cfg.max_lanes,
            )

            keep = keep[:num_to_keep]       # (N_keep, )
            scores = scores[keep]           # (N_keep, )
            lanes_preds = lanes_preds[keep]     # (N_keep, 3+S)
            labels = labels[keep]           # (N_keep, )

            if lanes_preds.shape[0] == 0:
                results_list.append([])
                continue

            pred = self.predictions_to_pred(scores, lanes_preds, labels, img_meta)
            results_list.append(pred)

        return results_list

    def predictions_to_pred(self, scores, lanes_preds, labels, img_metas=None):
        '''
        Convert predictions to internal Lane structure for evaluation.
        Args:
            scores: (N_pred, )
            lanes_preds: (N_pred, 3+S)
                3+S: 1 start_y(normalized), 1 start_x(absolute), 1 length(absolute), S coordinates(absolute)
            labels: (N_pred, )
        Returns:
            lanes: List[lane0, lane1, ...]
        '''
        ori_height, ori_weight = img_metas['ori_shape'][:2]
        crop = img_metas.get('crop', (0, 0, 0, 0))     # (x_min, y_min, x_max, y_max)
        y_min = crop[1]

        self.anchor_ys = self.anchor_ys.to(lanes_preds.device)    # (S, ),  [1, ..., 0]
        self.anchor_ys = self.anchor_ys.double()
        lanes = []
        for score, lane, label in zip(scores, lanes_preds, labels):
            # score: float
            # lane: (3+S, )  3: start_y, start_x, len
            lane_xs = lane[3:] / (self.img_w - 1)     # normalized value
            # start_y(normalized --> absolute)
            start = min(max(0, int(round(lane[0].item() * self.n_strips))),
                        self.n_strips)
            length = int(round(lane[2].item()))
            end = start + length - 1
            end = min(end, len(self.anchor_ys) - 1)

            # end = label_end
            # if the prediction does not start at the bottom of the image,
            # extend its prediction until the x is outside the image
            mask = ~((((lane_xs[:start] >= 0.) & (lane_xs[:start] <= 1.)
                       ).cpu().numpy()[::-1].cumprod()[::-1]).astype(np.bool))
            lane_xs[end + 1:] = -2
            lane_xs[:start][mask] = -2
            lane_ys = self.anchor_ys[lane_xs >= 0]   # (N_valid, )   由底向上
            lane_xs = lane_xs[lane_xs >= 0]         # (N_valid, )
            lane_xs = lane_xs.flip(0).double()      # (N_valid, )   由上向底
            lane_ys = lane_ys.flip(0)               # (N_valid, )   由上向底

            lane_ys = (lane_ys * (ori_height - y_min) + y_min) / ori_height
            if len(lane_xs) <= 1:
                continue
            points = torch.stack((lane_xs, lane_ys), dim=-1)  # (N_valid, 2)  normalized
            lane = Lane(points=points.cpu().numpy(),
                        metadata={
                            'start_x': lane[1],
                            'start_y': lane[0],
                            'conf': score,
                            'label': label
                        })
            lanes.append(lane)
        return lanes

