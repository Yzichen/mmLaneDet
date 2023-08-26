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


def LinearModule(hidden_dim):
    return nn.ModuleList(
        [nn.Linear(hidden_dim, hidden_dim),
         nn.ReLU(inplace=True)])


class FeatureResize(nn.Module):
    def __init__(self, size=(10, 25)):
        super(FeatureResize, self).__init__()
        self.size = size

    def forward(self, x):
        x = F.interpolate(x, self.size)
        return x.flatten(2)


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
                 num_priors,
                 sample_points,
                 fc_hidden_dim,
                 refine_layers,
                 mid_channels=48):
        super(ROIGather, self).__init__()
        self.in_channels = in_channels
        self.num_priors = num_priors
        self.f_key = ConvModule(in_channels=self.in_channels,
                                out_channels=self.in_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                conv_cfg=dict(type='Conv2d'),
                                norm_cfg=dict(type='BN'))

        self.f_query = nn.Sequential(
            nn.Conv1d(in_channels=num_priors,
                      out_channels=num_priors,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      groups=num_priors),
            # nn.Conv1d(in_channels=self.in_channels,
            #           out_channels=self.in_channels,
            #           kernel_size=1,
            #           stride=1,
            #           padding=0),
            nn.ReLU(),
        )

        self.f_value = nn.Conv2d(in_channels=self.in_channels,
                                 out_channels=self.in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.W = nn.Conv1d(in_channels=num_priors,
                           out_channels=num_priors,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=num_priors)

        self.resize = FeatureResize()
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

        self.convs = nn.ModuleList()
        self.catconv = nn.ModuleList()
        for i in range(refine_layers):
            self.convs.append(
                ConvModule(in_channels,
                           mid_channels, (9, 1),
                           padding=(4, 0),
                           bias=False,
                           norm_cfg=dict(type='BN')))

            self.catconv.append(
                ConvModule(mid_channels * (i + 1),
                           in_channels, (9, 1),
                           padding=(4, 0),
                           bias=False,
                           norm_cfg=dict(type='BN')))

        self.fc = nn.Linear(sample_points * fc_hidden_dim, fc_hidden_dim)
        self.fc_norm = nn.LayerNorm(fc_hidden_dim)

    def roi_fea(self, x, layer_index):
        """
        Args:
            x: List [(B*num_priors, C, N_sample, 1),  ...]   len=(layer_index+1)
            layer_index: int
        Returns:
            cat_feat: (B*num_priors, C, N_sample, 1)
        """
        feats = []
        for i, feature in enumerate(x):
            # feat_trans: (B*num_priors, C, N_sample, 1) --> (B*num_priors, C_mid, N_sample, 1)
            feat_trans = self.convs[i](feature)
            feats.append(feat_trans)

        # (B*num_priors, C_mid*(layer_index+1), N_sample, 1)
        cat_feat = torch.cat(feats, dim=1)
        # (B*num_priors, C_mid*(layer_index+1), N_sample, 1) --> (B*num_priors, C, N_sample, 1)
        cat_feat = self.catconv[layer_index](cat_feat)
        return cat_feat

    def forward(self, roi_features, x, layer_index):
        '''
        Args:
            roi_features: prior feature, List [(B*num_priors, C, N_sample, 1),  ...], 对应不同FPN level.
            x: feature map  (B, C, fH, fW)
            layer_index: currently on which layer to refine
        Return:
            roi: prior features with gathered global information, shape: (B, num_priors, C)
        '''
        roi = self.roi_fea(roi_features, layer_index)   # (B*num_priors, C, N_sample, 1)
        bs = x.size(0)
        roi = roi.contiguous().view(bs * self.num_priors, -1)   # (B*num_priors, C*N_sample)
        # (B*num_priors, C*N_sample) --> (B*num_priors, C)
        roi = F.relu(self.fc_norm(self.fc(roi)))
        # (B*num_priors, C) --> (B, num_priors, C)
        roi = roi.view(bs, self.num_priors, -1)

        query = roi     # (B, num_priors, C)
        # (B, num_priors, C) --> (B, num_priors, C)
        query = self.f_query(query)
        # query = self.f_query(query.permute(0, 2, 1)).permute(0, 2, 1)

        # (B, C, fH, fW) --> (B, C, fH, fW) --> (B, C, 10*25)
        value = self.resize(self.f_value(x))
        # (B, C, fH, fW) --> (B, C, fH, fW) --> (B, C, 10*25)
        key = self.resize(self.f_key(x))

        # (B, C, 10*25) --> (B, 10*25, C)
        value = value.permute(0, 2, 1)

        # (B, num_priors, C) @ (B, C, 10*25) --> (B, num_priors, 10*25)
        sim_map = torch.matmul(query, key)
        sim_map = (self.in_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # (B, num_priors, 10*25) @ (B, 10*25, C) --> (B, num_priors, C)
        context = torch.matmul(sim_map, value)
        # (B, num_priors, C) --> (B, num_priors, C)
        context = self.W(context)

        # (B, num_priors, C) + (B, num_priors, C) --> (B, num_priors, C)
        roi = roi + F.dropout(context, p=0.1, training=self.training)

        return roi


class SegDecoder(nn.Module):
    '''
    Optionaly seg decoder
    '''
    def __init__(self,
                 num_class,
                 prior_feat_channels=64,
                 refine_layers=3):
        super().__init__()
        self.dropout = nn.Dropout2d(0.1)
        self.conv = nn.Conv2d(prior_feat_channels * refine_layers, num_class, 1)

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


@HEADS.register_module
class CLRHead(BaseDenseHead):
    def __init__(self,
                 num_points=72,
                 sample_points=36,
                 prior_feat_channels=64,
                 fc_hidden_dim=64,
                 num_priors=192,
                 num_fc=2,
                 refine_layers=3,
                 num_classes=1,
                 seg_num_classes=4,
                 img_size=(800, 320),
                 code_weights=None,
                 loss_cls=dict(
                     type='FocalLoss',
                     gamma=2.0,
                     alpha=0.25,
                     use_sigmoid=True,
                     reduction='mean',
                     loss_weight=6
                 ),
                 loss_lane=dict(
                     type='SmoothL1Loss',
                     beta=1.0,
                     reduction='mean',
                     loss_weight=0.5
                 ),
                 loss_iou=dict(
                     type='LineIou_Loss',
                     length=15,
                     reduction='mean',
                     loss_weight=2.0
                 ),
                 loss_seg=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     ignore_index=255,
                     class_weight=1.0,
                     bg_cls_weight=0.4,
                     loss_weight=1.0
                 ),
                 train_cfg=None,
                 test_cfg=None,
                 sync_cls_avg_factor=False,
                 with_seg=True,
                 init_cfg=None,
                 **kwargs):
        super(CLRHead, self).__init__(init_cfg)
        self.n_strips = num_points - 1
        self.n_offsets = num_points
        self.num_priors = num_priors
        self.sample_points = sample_points
        self.refine_layers = refine_layers
        self.fc_hidden_dim = fc_hidden_dim
        self.num_fc = num_fc
        self.img_w, self.img_h = img_size
        self.num_classes = num_classes
        self.seg_num_classes = seg_num_classes

        # 从num_points个点中采样sample_points个点，来pooling lane priors的特征.
        self.register_buffer(name='sample_x_indexs', tensor=(torch.linspace(
            0, 1, steps=self.sample_points, dtype=torch.float32) *
                                self.n_strips).long())
        self.register_buffer(name='prior_feat_ys', tensor=torch.flip(
            (1 - self.sample_x_indexs.float() / self.n_strips), dims=[-1]))
        self.register_buffer(name='prior_ys', tensor=torch.linspace(1, 0, steps=self.n_offsets, dtype=torch.float32))

        self.prior_feat_channels = prior_feat_channels
        # 初始化lane prior 来生成self.prior_embeddings (num_priors, 3)  3: (start_y, start_x, theta)
        self._init_prior_embeddings()

        # priors: (num_priors, 4 + S)
        # priors_on_featmap: (num_priors, N_sample)
        init_priors, priors_on_featmap = self.generate_priors_from_embeddings()
        self.register_buffer(name='priors', tensor=init_priors)
        self.register_buffer(name='priors_on_featmap', tensor=priors_on_featmap)

        self.roi_gather = ROIGather(self.prior_feat_channels, self.num_priors,
                                    self.sample_points, self.fc_hidden_dim,
                                    self.refine_layers)

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor

        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is CLRHead):
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
        self.loss_lane = build_loss(loss_lane)
        self.loss_iou = build_loss(loss_iou)

        if train_cfg:
            assigner = train_cfg['assigner']
            self.assigner = build_assigner(assigner)
            self.assigner.img_w = self.img_w
            self.assigner.img_h = self.img_h

        self.test_cfg = test_cfg

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
                                          self.refine_layers)

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

        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0]     # start_y, start_x, theta, length
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)

        self._init_layers()
        # init the weights here
        self.init_weights()

    def _init_layers(self):
        reg_modules = list()
        cls_modules = list()
        for _ in range(self.num_fc):
            reg_modules += [*LinearModule(self.fc_hidden_dim)]
            cls_modules += [*LinearModule(self.fc_hidden_dim)]
        self.reg_modules = nn.ModuleList(reg_modules)
        self.cls_modules = nn.ModuleList(cls_modules)

        self.reg_layers = nn.Linear(
            self.fc_hidden_dim, self.n_offsets + 1 + 2 +
            1)  # n offsets + 1 length + start_x + start_y + theta
        self.cls_layers = nn.Linear(self.fc_hidden_dim, self.cls_out_channels)

    # function to init layer weights
    def init_weights(self):
        # initialize heads
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.cls_layers.bias, bias_init)
        else:
            for m in self.cls_layers.parameters():
                nn.init.normal_(m, mean=0., std=1e-3)

        for m in self.reg_layers.parameters():
            nn.init.normal_(m, mean=0., std=1e-3)

    def _init_prior_embeddings(self):
        # [start_y, start_x, theta] -> all normalize
        self.prior_embeddings = nn.Embedding(self.num_priors, 3)

        bottom_priors_nums = self.num_priors * 3 // 4
        left_priors_nums, _ = self.num_priors // 8, self.num_priors // 8

        strip_size = 0.5 / (left_priors_nums // 2 - 1)
        bottom_strip_size = 1 / (bottom_priors_nums // 4 + 1)
        for i in range(left_priors_nums):
            nn.init.constant_(self.prior_embeddings.weight[i, 0],
                              (i // 2) * strip_size)
            nn.init.constant_(self.prior_embeddings.weight[i, 1], 0.)
            nn.init.constant_(self.prior_embeddings.weight[i, 2],
                              0.16 if i % 2 == 0 else 0.32)

        for i in range(left_priors_nums,
                       left_priors_nums + bottom_priors_nums):
            nn.init.constant_(self.prior_embeddings.weight[i, 0], 0.)
            nn.init.constant_(self.prior_embeddings.weight[i, 1],
                              ((i - left_priors_nums) // 4 + 1) *
                              bottom_strip_size)
            nn.init.constant_(self.prior_embeddings.weight[i, 2],
                              0.2 * (i % 4 + 1))

        for i in range(left_priors_nums + bottom_priors_nums, self.num_priors):
            nn.init.constant_(
                self.prior_embeddings.weight[i, 0],
                ((i - left_priors_nums - bottom_priors_nums) // 2) *
                strip_size)
            nn.init.constant_(self.prior_embeddings.weight[i, 1], 1.)
            nn.init.constant_(self.prior_embeddings.weight[i, 2],
                              0.68 if i % 2 == 0 else 0.84)

    def generate_priors_from_embeddings(self):
        """
        Returns:
            priors: (num_priors, 4+S)
                4+S: 1 start_y(normalized), 1 start_x(normalized), 1 theta, 1 length, 72 coordinates(normalized)
            priors_on_featmap: (num_priors, N_sample)
        """
        predictions = self.prior_embeddings.weight  # (num_priors, 3)

        # 1 start_y, 1 start_x, 1 theta, 1 length, 72 coordinates,
        # (num_priors, 4+S)
        priors = predictions.new_zeros(
            (self.num_priors, 2 + 2 + self.n_offsets), device=predictions.device)

        priors[:, 0:3] = predictions.clone()    # 1 start_y, 1 start_x, 1 theta
        # self.prior_ys: [1, ..., 0]  len=S   从底向上排列
        # start_x + ((1 - prior_y) - start_y) / tan(theta)     normalized x coords   按照图像底部-->顶部排列.
        priors[:, 4:] = (
            priors[:, 1].unsqueeze(1).clone().repeat(1, self.n_offsets) *
            (self.img_w - 1) +
            ((1 - self.prior_ys.repeat(self.num_priors, 1) -
              priors[:, 0].unsqueeze(1).clone().repeat(1, self.n_offsets)) *
             self.img_h / torch.tan(priors[:, 2].unsqueeze(1).clone().repeat(
                 1, self.n_offsets) * math.pi + 1e-5))) / (self.img_w - 1)

        # 在lane priors上采样sample_points个点，用于聚合lane priors在feature map上的特征
        # (num_priors, N_sample)        按照图像底部-->顶部排列.
        priors_on_featmap = priors.clone()[..., 4 + self.sample_x_indexs]
        return priors, priors_on_featmap

    def draw_anchors(self, num=50):
        # for vis
        img = np.ones((self.img_h, self.img_w, 3), dtype=np.uint8) * 255
        idx = np.random.randint(0, len(self.priors), num)
        priors = self.priors[idx]

        for idx, prior in enumerate(priors):
            x_s = prior[4:] * (self.img_w - 1)
            y_s = self.prior_ys * self.img_h

            for i in range(1, self.n_offsets):
                cv2.line(img, pt1=(x_s[i-1].int().item(), y_s[i-1].int().item()),
                         pt2=(x_s[i].int().item(), y_s[i].int().item()), color=(255, 0, 0), thickness=2,
                         lineType=cv2.LINE_AA)

        cv2.imwrite("anchors.png", img)

    def pool_prior_features(self, batch_features, num_priors, prior_xs):
        '''
        pool prior feature from feature map.
        Args:
            batch_features (Tensor): Input feature maps, shape: (B, C, fH, fW)
            num_priors: int
            prior_xs: (B, num_priors, N_sample)
        Returns:
            feature: (B*num_priors, C, N_sample, 1)
        '''

        batch_size = batch_features.shape[0]

        # (B, num_priors, N_sample， 1)
        prior_xs = prior_xs.view(batch_size, num_priors, -1, 1)
        # (N_sample, ) --> (B*N_sample*num_priors, ) --> (B, num_priors, N_sample, 1)
        prior_ys = self.prior_feat_ys.repeat(batch_size * num_priors).view(
            batch_size, num_priors, -1, 1)

        # (0, 1) --> (-1, 1)
        prior_xs = prior_xs * 2. - 1.
        prior_ys = prior_ys * 2. - 1.
        grid = torch.cat((prior_xs, prior_ys), dim=-1)      # (B, num_priors, N_sample, 2)
        # (B, C, fH, fW) --> (B, C, num_priors, N_sample) --> (B, C, num_priors, N_sample)
        # --> (B, num_priors, C, N_sample)
        feature = F.grid_sample(batch_features, grid,
                                align_corners=True).permute(0, 2, 1, 3)

        # (B*num_priors, C, N_sample, 1)
        feature = feature.reshape(batch_size * num_priors,
                                  self.prior_feat_channels, self.sample_points,
                                  1)

        return feature

    # forward function here
    def forward(self, x, **kwargs):
        '''
        Take pyramid features as input to perform Cross Layer Refinement and finally output the prediction lanes.
        Each feature is a 4D tensor.
        Args:
            x: input features (list[Tensor])
            kwargs:
                dict {
                    'img': (B, 3, img_H, img_W),
                    'gt_lanes': (B, max_lanes, 4+S),
                        4+S: 1 start_y (normalized), 1 start_x (absolute), 1 theta, 1 length (absolute),
                             S coordinates(absolute)
                    'seg': (B, img_H, img_W),
                    'meta': List[dict0, dict1, ...]
                }
        Return:
            prediction_list: each layer's prediction result
            seg: segmentation result for auxiliary loss
        '''
        batch_features = list(x[len(x) - self.refine_layers:])
        # high level --> low level
        batch_features.reverse()    # List[(B, C, H5, W5), (B, C, H4, W4), (B, C, H3, W3)]
        batch_size = batch_features[-1].shape[0]    # batch_size

        if self.training:
            # 由于priors是可学习的，因此训练过程中一直在变化.
            # priors: (num_priors, 4+S)
            # priors_on_featmap: (num_priors, N_sample)
            self.priors, self.priors_on_featmap = self.generate_priors_from_embeddings()

        # priors: (B, num_priors, 4+S)  1 start_y, 1 start_x, 1 theta, 1 length, 72 coordinates,
        # priors_on_featmap: (B, num_priors, N_sample)
        priors = self.priors.repeat(batch_size, 1, 1)
        priors_on_featmap = self.priors_on_featmap.repeat(batch_size, 1, 1)

        # iterative refine
        outputs_classes = []
        outputs_lanes = []
        prior_features_stages = []
        for stage in range(self.refine_layers):
            # print(batch_features[stage].shape, batch_features[stage].sum())
            num_priors = priors_on_featmap.shape[1]
            prior_xs = torch.flip(priors_on_featmap, dims=[2])      # 图像顶部-->底部

            batch_prior_features = self.pool_prior_features(
                batch_features[stage], num_priors, prior_xs)    # (B*num_priors, C, N_sample, 1)
            prior_features_stages.append(batch_prior_features)

            fc_features = self.roi_gather(prior_features_stages,
                                          batch_features[stage], stage)     # (B, num_priors, C)

            # (B, num_priors, C) --> (B*num_priors, C)
            fc_features = fc_features.view(batch_size * num_priors, self.fc_hidden_dim)

            cls_features = fc_features.clone()
            reg_features = fc_features.clone()
            for cls_layer in self.cls_modules:
                # (B*num_priors, C)
                cls_features = cls_layer(cls_features)
            for reg_layer in self.reg_modules:
                # (B*num_priors, C)
                reg_features = reg_layer(reg_features)

            # (B*num_priors, C) --> (B*num_priors, 2)
            cls_logits = self.cls_layers(cls_features)
            # (B*num_priors, C) --> (B*num_priors, 2+1+1+n_offsets)
            reg = self.reg_layers(reg_features)

            cls_logits = cls_logits.reshape(
                batch_size, -1, cls_logits.shape[1])  # (B, num_priors, n_cls)
            outputs_classes.append(cls_logits)

            reg = reg.reshape(batch_size, -1, reg.shape[1])    # (B, num_priors, 2+1+1+n_offsets)
            predictions = priors.clone()    # (B, num_priors, 4+S)
            # start_y, 1 start_x, 1 theta   (B, num_priors, 3)
            predictions[:, :, 0:3] += reg[:, :, :3]    # also reg theta angle here
            # Todo 这里加上sigmoid会不会更好
            predictions[:, :, 3] = reg[:, :, 3]      # length   (B, num_priors), 这似乎是一个normalized length.

            def tran_tensor(t):
                return t.unsqueeze(2).clone().repeat(1, 1, self.n_offsets)

            # (B, num_priors, S)
            predictions[..., 4:] = (
                tran_tensor(predictions[..., 1]) * (self.img_w - 1) +
                ((1 - self.prior_ys.repeat(batch_size, num_priors, 1) -
                  tran_tensor(predictions[..., 0])) * self.img_h /
                 torch.tan(tran_tensor(predictions[..., 2]) * math.pi + 1e-5))) / (self.img_w - 1)

            # 更新后的line priors(考虑start_x、start_y、theta和length的更新)，作为下一个layer的line prior.
            prediction_lines = predictions.clone()      # (B, num_priors, 4+S)
            # (B, num_priors, S)
            predictions[..., 4:] += reg[..., 4:]
            outputs_lanes.append(predictions)

            if stage != self.refine_layers - 1:
                priors = prediction_lines.detach().clone()
                priors_on_featmap = priors[..., 4+self.sample_x_indexs]

        all_cls_scores = torch.stack(outputs_classes)   # (num_layers, B, num_priors, 2)
        all_lanes_preds = torch.stack(outputs_lanes)    # (num_layers, B, num_priors, 4+S)

        output = {
            'all_cls_scores': all_cls_scores,   # (num_layers, B, num_priors, 2)
            'all_lanes_preds': all_lanes_preds,     # (num_layers, B, num_priors, 4+S)
        }
        if self.training:
            if self.with_seg:
                # (B, refine_layers*C, H3, W3)
                seg_features = torch.cat([
                    F.interpolate(feature,
                                  size=[
                                      batch_features[-1].shape[2],
                                      batch_features[-1].shape[3]
                                  ],
                                  mode='bilinear',
                                  align_corners=False)
                    for feature in batch_features
                ], dim=1)
                # (B, refine_layers*C, H3, W3)  --> (B, num_class, img_H, img_W)
                seg = self.seg_decoder(seg_features, self.img_h, self.img_w)
                output['pred_seg'] = seg

        return output

    @force_fp32(apply_to=('preds_dicts'))
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
                'all_cls_scores': (num_layers, B, num_priors, n_cls)
                'all_lanes_preds':  (num_layers, B, num_priors, 4+S)
                    4+S: 1 start_y(normalized), 1 start_x(normalized), 1 theta, 1 length(normalized),
                       S coordinates(normalized),
                'seg': (B, num_class, img_H, img_W)
            }
            gt_lanes (List):  List[(N_lanes0, 4+S), (N_lanes1, 4+S), ...]
                4+S: 1 start_y (normalized), 1 start_x (absolute), 1 theta, 1 length (absolute),
                     S coordinates(absolute)
            gt_labels (List): List[(N_lanes0, ), (N_lanes1, ), ...]
            gt_semantic_seg (Tensor): (B, 1, H, W)
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # List[(B, num_priors, n_cls), (B, num_priors, n_cls), (B, num_priors, n_cls)]
        all_cls_scores = preds_dicts['all_cls_scores']
        # List[(B, num_priors, 4+S), (B, num_priors, 4+S), (B, num_priors, 4+S)]
        all_lanes_preds = preds_dicts['all_lanes_preds']

        # List[List[(N_lanes0, 4+S), (N_lanes1, 4+S), ...], List[(N_lanes0, 4+S), (N_lanes1, 4+S), ...], ...]
        all_gt_lanes_list = [gt_lanes for _ in range(self.refine_layers)]

        # List[List[(N_lanes0, ), (N_lanes1, ), ...], List[(N_lanes0, ), (N_lanes1, ), ...], ...]
        all_gt_labels_list = [gt_labels for _ in range(self.refine_layers)]

        # 分别计算每层layer的loss
        losses_cls, losses_yxtl, losses_iou, cls_accuracy_list = multi_apply(
            self.loss_single, all_cls_scores, all_lanes_preds, all_gt_lanes_list, all_gt_labels_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_yxtl'] = losses_yxtl[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        loss_dict['cls_accuracy'] = cls_accuracy_list[-1]

        # loss from other decoder layers
        num_dec_layers = 0
        for loss_cls_i, loss_yxtl_i, loss_iou_i, cls_accuracy_i in zip(losses_cls[:-1], losses_yxtl[:-1],
                                                                 losses_iou[:-1], cls_accuracy_list[:-1]):
            loss_dict[f'd{num_dec_layers}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layers}.loss_yxtl'] = loss_yxtl_i
            loss_dict[f'd{num_dec_layers}.loss_iou'] = loss_iou_i
            loss_dict[f'd{num_dec_layers}.cls_accuracy'] = cls_accuracy_i
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

    def loss_single(self, cls_scores, lanes_preds, gt_lanes, gt_labels):
        """
        :param
            cls_scores: (B, num_priors, n_cls)
            lanes_preds:  (B, num_priors, 4+S)
                4+S: 1 start_y(normalized), 1 start_x(normalized), 1 theta(normalized), 1 length(normalized),
                     S coordinates(normalized),
            gt_lanes:  List[(N_lanes0, 4+S), (N_lanes1, 4+S), ...]
                4+S: 1 start_y (normalized), 1 start_x (absolute), 1 theta(normalized), 1 length (absolute),
                     S coordinates(absolute)
            gt_labels: List[(N_lanes0, ), (N_lanes1, ), ...]
        :return:
        """
        num_imgs = cls_scores.shape[0]
        # List[(num_priors, n_cls), (num_priors, n_cls), ...]
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        # List[(num_priors, 4+S), (num_priors, 4+S), ...]
        lane_preds_list = [lanes_preds[i] for i in range(num_imgs)]

        num_total_gt_lanes = sum([gt_lane.shape[0] for gt_lane in gt_lanes])

        cls_reg_targets = self.get_targets(
            cls_scores_list,    # List[(num_priors, n_cls), (num_priors, n_cls), ...]    len = batch_size
            lane_preds_list,    # List[(num_priors, 4+S), (num_priors, 4+S), ...]   len = batch_size
            gt_lanes,       # List[(N_gt0, 4+S), (N_gt1, 4+S), ...]     len = batch_size
            gt_labels       # List[(N_gt0,), (N_gt1,), ...]
        )

        # cls_reg_targets:
        # labels_list: List[(num_priors,), (num_priors,), ...]
        # label_weights_list: List[(num_priors,), (num_priors,), ...]
        # lane_targets_list: List[(num_priors, 4 + S), (num_priors, 4 + S), ...]
        # lane_weights_list: List[(num_priors, 4 + S), (num_priors, 4 + S), ...]
        # pos_inds_list: int    N_pos0+N_pos1+...
        # neg_inds_list: int    N_neg0+N_neg1+...

        (labels_list, label_weights_list, lane_targets_list, lane_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)      # (B*num_priors, )
        label_weights = torch.cat(label_weights_list, 0)    # (B*num_priors, )
        lane_targets = torch.cat(lane_targets_list, 0)      # (B*num_priors, 4+S)
        lane_weights = torch.cat(lane_weights_list, 0)      # (B*num_priors, 4+S)

        # 1. classification loss
        cls_scores = cls_scores.view(-1, cls_scores.shape[-1])      # (B*num_priors, n_cls)
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1.0)

        loss_cls = self.loss_cls(
            cls_scores,      # (B*num_priors, n_cls)
            labels,          # (B*num_priors, )
            label_weights,   # (B*num_priors, )
            avg_factor=cls_avg_factor)

        # 2. regression loss
        lanes_preds = lanes_preds.view(-1, lanes_preds.size(-1))  # (B*num_priors, 4+S)
        # 2.1  yxtl regression loss
        # (B*num_priors, 4)
        # start_y(normalized), start_x(normalized), 1 theta(normalized), 1 length(normalized)
        # normalized --> absolute
        pred_yxtl = lanes_preds[..., :4].clone()
        pred_yxtl[:, 0] *= self.n_strips
        pred_yxtl[:, 1] *= (self.img_w - 1)
        pred_yxtl[:, 2] *= 180
        pred_yxtl[:, 3] *= self.n_strips

        target_yxtl = lane_targets[..., :4].clone()
        # (B*num_priors, 4)
        # start_y(normalized), start_x(absolute), 1 theta(normalized), 1 length(absolute)
        # normalized --> absolute
        target_yxtl[:, 0] *= self.n_strips
        target_yxtl[:, 2] *= 180

        # 调整length
        with torch.no_grad():
            pred_start_y = torch.clamp(pred_yxtl[:, 0].round().long(), 0, self.n_strips)
            target_start_y = target_yxtl[:, 0].round().long()
            target_yxtl[:, -1] -= (pred_start_y - target_start_y)

        loss_yxtl = self.loss_lane(
            pred_yxtl,
            target_yxtl,
            weight=lane_weights[..., :4] * self.code_weights,
            avg_factor=num_total_pos * 4,
        )

        # 2.2  line Iou loss (based on x_coords)
        # (B*num_priors, S)
        # normalized --> absolute
        pred_x_coords = lanes_preds[..., 4:].clone() * (self.img_w - 1)
        # (B*num_priors, S)  absolute
        target_x_coords = lane_targets[..., 4:]

        loss_iou = self.loss_iou(
            pred_x_coords,
            target_x_coords,
            self.img_w,
            weight=lane_weights[..., 0],
            avg_factor=num_total_pos,
        )

        # calculate acc
        cls_accuracy = accuracy(cls_scores, labels)
        return loss_cls, loss_yxtl, loss_iou, cls_accuracy

    def get_targets(self, cls_scores_list, lane_preds_list, gt_lanes_list, gt_labels_list):
        """
        :param cls_scores_list:     # List[(num_priors, n_cls), (num_priors, n_cls), ...]
        :param lane_preds_list:     # List[(num_priors, 4+S), (num_priors, 4+S), ...]
        :param gt_lanes_list:       # List[(N_gt0, 4+S), (N_gt1, 4+S), ...]
        :param gt_labels_list:      # List[(N_gt0, ), (N_gt1, ), ...]
        :return:
            labels_list: List[(num_priors, ), (num_priors, ), ...]
            label_weights_list: List[(num_priors, ), (num_priors, ), ...]
            lane_targets_list: List[(num_priors, 4+S), (num_priors, 4+S), ...]
            lane_weights_list: List[(num_priors, 4+S), (num_priors, 4+S), ...]
        """
        (labels_list, label_weights_list, lane_targets_list, lane_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, lane_preds_list, gt_lanes_list, gt_labels_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, lane_targets_list, lane_weights_list,
                num_total_pos, num_total_neg)

    def _get_target_single(self,
                           cls_scores,
                           lane_preds,
                           gt_lanes,
                           gt_labels):
        """"
        Args:
            cls_scores (Tensor): (num_priors, n_cls)
            lane_preds (Tensor): (num_priors, 4+S)
            gt_lanes (Tensor): (N_gt, 4+S)
            gt_labels (Tensor): (N_gt, )
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - lane_targets (Tensor): Lane targets of each image.
                - lane_weights (Tensor): Lane weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        num_priors = lane_preds.size(0)
        # assigner
        # (num_prior, ),  (num_prior, )
        with torch.no_grad():
            assigned_gt_inds, assigned_labels = self.assigner.assign(cls_scores, lane_preds, gt_lanes, gt_labels)

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
            pos_gt_lanes = gt_lanes[pos_assigned_gt_inds.long(), :]   # (N_pos, 4+S)

        # label targets
        # 默认为bg
        labels = cls_scores.new_full((num_priors,), fill_value=self.num_classes, dtype=torch.long)
        label_weights = cls_scores.new_zeros((num_priors, ), dtype=torch.float)
        # lane targets
        # (num_priors, 4+S)
        lane_targets = torch.zeros_like(lane_preds)
        lane_weights = torch.zeros_like(lane_preds)

        if len(pos_inds) > 0:
            labels[pos_inds] = pos_gt_labels
            label_weights[pos_inds] = 1.0
            lane_targets[pos_inds] = pos_gt_lanes  # (N_pos, 4+S)
            lane_weights[pos_inds] = 1.0

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        assert label_weights.all(), "label_weights 有问题, 分类时应当考虑所有样本."
        return (labels, label_weights, lane_targets, lane_weights,
                pos_inds, neg_inds)

    @force_fp32(apply_to=('preds_dicts'))
    def get_lanes(self, preds_dicts, img_metas, cfg=None):
        '''
        Convert model output to lanes.
        Args:
            preds_dicts = dict {
                'all_cls_scores': all_cls_scores,   # (num_layers, B, num_priors, 2)
                'all_lanes_preds': all_lanes_preds,     # (num_layers, B, num_priors, 4+S)
                    4+S:  1 start_y(normalized), 1 start_x(normalized), 1 theta(normalized),
                          1 length(normalized), S coordinates(normalized)
            }
        Returns:
            results_list: Lane results of each image. The outer list corresponds to each image. The inner list
                corresponds to each lane.
                List[List[lane0, lane1, ...], ...]
        '''
        output_cls_scores = preds_dicts['all_cls_scores'][-1]   # (B, num_priors, n_cls)
        output_lanes_preds = preds_dicts['all_lanes_preds'][-1]     # (B, num_priors, 4+S)

        cfg = self.test_cfg if cfg is None else cfg
        nms_pre = cfg.get('nms_pre', self.num_priors)

        results_list = []
        for cls_scores, lanes_preds, img_meta in zip(output_cls_scores, output_lanes_preds, img_metas):
            # cls_scores: (num_priors, n_cls)
            # lanes_preds: (num_priors, 4+S)
            if self.use_sigmoid_cls:
                scores = cls_scores.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_scores.softmax(-1)[:, :-1]

            # filter out the conf lower than conf threshold
            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre, dict(lanes_preds=lanes_preds)
            )
            # scores: (N, )    # N = nms_pre if num_all_valid_anchors > nms_pre else num_all_valid_anchors
            # labels: (N, )
            # keep_idxs: (N, )
            # filtered_results: {'lanes_preds': (N, 4+S)}
            scores, labels, keep_idxs, filtered_results = results
            lanes_preds = filtered_results['lanes_preds']   # (N, 4+S)

            if lanes_preds.shape[0] == 0:
                results_list.append([])
                continue

            scores = scores.detach().clone()
            lanes_preds = lanes_preds.detach().clone()
            # (N, 3+S)   1 start_y(normalized), 1 start_x(normalized), 1 length(normalized), S coordinates(normalized)
            nms_predictions = torch.cat(
                [lanes_preds[..., :2], lanes_preds[..., 3:]], dim=-1)
            # normalized length --> absolute length
            nms_predictions[..., 2] = nms_predictions[..., 2] * self.n_strips
            # normalized x_coords --> absolute x_coords
            nms_predictions[...,
                            3:] = nms_predictions[..., 3:] * (self.img_w - 1)

            keep, num_to_keep, _ = nms(
                nms_predictions,    # (N, 3+S)
                scores,             # (N, )
                overlap=cfg.nms_thres,
                top_k=cfg.max_lanes,
            )

            keep = keep[:num_to_keep]       # (N_keep, )
            scores = scores[keep]           # (N_keep, )
            lanes_preds = lanes_preds[keep]     # (N_keep, 4+S)
            labels = labels[keep]           # (N_keep, )

            if lanes_preds.shape[0] == 0:
                results_list.append([])
                continue

            # normalized length --> absolute length
            lanes_preds[:, 3] = torch.round(lanes_preds[:, 3] * self.n_strips)
            pred = self.predictions_to_pred(scores, lanes_preds, labels, img_meta)
            results_list.append(pred)

        return results_list

    def predictions_to_pred(self, scores, lanes_preds, labels, img_metas=None):
        '''
        Convert predictions to internal Lane structure for evaluation.
        Args:
            scores: (N_pred, )
            lanes_preds: (N_pred, 4+S)
                4+S: 1 start_y(normalized), 1 start_x(normalized), 1 theta(normalized), 1 length(absolute), S coordinates(normalized)
            labels: (N_pred, )
        Returns:
            lanes: List[lane0, lane1, ...]
        '''
        ori_height, ori_weight = img_metas['ori_shape'][:2]
        crop = img_metas.get('crop', (0, 0, 0, 0))     # (x_min, y_min, x_max, y_max)
        y_min = crop[1]

        self.prior_ys = self.prior_ys.to(lanes_preds.device)
        self.prior_ys = self.prior_ys.double()
        lanes = []
        for score, lane, label in zip(scores, lanes_preds, labels):
            # score: float
            # lane: (4+S, )
            lane_xs = lane[4:]      # normalized value
            # start_y(normalized --> absolute)
            start = min(max(0, int(round(lane[0].item() * self.n_strips))),
                        self.n_strips)
            length = int(round(lane[3].item()))
            end = start + length - 1
            end = min(end, len(self.prior_ys) - 1)
            # end = label_end
            # if the prediction does not start at the bottom of the image,
            # extend its prediction until the x is outside the image
            mask = ~((((lane_xs[:start] >= 0.) & (lane_xs[:start] <= 1.)
                       ).cpu().numpy()[::-1].cumprod()[::-1]).astype(np.bool))
            lane_xs[end + 1:] = -2
            lane_xs[:start][mask] = -2
            lane_ys = self.prior_ys[lane_xs >= 0]   # (N_valid, )   由底向上
            lane_xs = lane_xs[lane_xs >= 0]         # (N_valid, )
            lane_xs = lane_xs.flip(0).double()      # (N_valid, )   由上向底
            lane_ys = lane_ys.flip(0)               # (N_valid, )   由上向底

            lane_ys = (lane_ys * (ori_height - y_min) + y_min) / ori_height
            if len(lane_xs) <= 2:
                continue
            points = torch.stack(
                (lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)),
                dim=1).squeeze(2)      # (N_valid, 2)  normalized
            lane = Lane(points=points.cpu().numpy(),
                        metadata={
                            'start_x': lane[1],
                            'start_y': lane[0],
                            'conf': score,
                            'label': label
                        })
            lanes.append(lane)
        return lanes