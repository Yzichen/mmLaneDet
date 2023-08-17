import torch
import torch.nn as nn
import torch.nn.functional as F

from mmlane.models import AGGREGATORS
from mmcv.runner import BaseModule, auto_fp16
from mmcv.cnn import ConvModule


@AGGREGATORS.register_module
class RESA(BaseModule):
    def __init__(self, in_channels=512, out_channels=128, mid_channels=None, kernel_size=9,
                directions=['d', 'u', 'r', 'l'], alpha=2.0, num_iters=4, img_size=(800, 320),
                stride=8):
        super(RESA, self).__init__()
        self.num_iters = num_iters
        self.alpha = alpha
        self.directions = directions
        self.kernel_size = kernel_size
        self.width = img_size[0] // stride
        self.height = img_size[1] // stride

        # if mid_channels is not None:
        #     self.channel_reducer = nn.Sequential(
        #         ConvModule(in_channels=in_channels,
        #                    out_channels=mid_channels,
        #                    kernel_size=3,
        #                    stride=1,
        #                    padding=4,
        #                    dilation=4,
        #                    conv_cfg=dict(type='Conv2d'),
        #                    norm_cfg=dict(type='BN'),
        #                    act_cfg=dict(type='ReLU')
        #                    ),
        #         ConvModule(in_channels=mid_channels,
        #                    out_channels=out_channels,
        #                    kernel_size=1,
        #                    stride=1,
        #                    conv_cfg=dict(type='Conv2d'),
        #                    norm_cfg=dict(type='BN'),
        #                    act_cfg=dict(type='ReLU')
        #                    )
        #     )
        # else:
        #     self.channel_reducer = ConvModule(in_channels=in_channels,
        #                                       out_channels=out_channels,
        #                                       kernel_size=1,
        #                                       stride=1,
        #                                       conv_cfg=dict(type='Conv2d'),
        #                                       norm_cfg=dict(type='BN'),
        #                                       act_cfg=dict(type='ReLU')
        #                                       )
        self.channel_reducer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                         bias=False)

        for i in range(self.num_iters):
            conv_d = nn.Conv2d(out_channels, out_channels, (1, kernel_size), padding=(0, kernel_size//2), bias=False)
            conv_u = nn.Conv2d(out_channels, out_channels, (1, kernel_size), padding=(0, kernel_size//2), bias=False)
            conv_r = nn.Conv2d(out_channels, out_channels, (kernel_size, 1), padding=(kernel_size//2, 0), bias=False)
            conv_l = nn.Conv2d(out_channels, out_channels, (kernel_size, 1), padding=(kernel_size//2, 0), bias=False)
            setattr(self, 'conv_d' + str(i), conv_d)
            setattr(self, 'conv_u' + str(i), conv_u)
            setattr(self, 'conv_r' + str(i), conv_r)
            setattr(self, 'conv_l' + str(i), conv_l)

            idx_d = (torch.arange(self.height) + self.height // 2**(self.num_iters - i)) % self.height
            idx_u = (torch.arange(self.height) - self.height // 2**(self.num_iters - i)) % self.height
            idx_r = (torch.arange(self.width) + self.width // 2**(self.num_iters - i)) % self.width
            idx_l = (torch.arange(self.width) - self.width // 2**(self.num_iters - i)) % self.width
            setattr(self, 'idx_d' + str(i), idx_d)
            setattr(self, 'idx_u' + str(i), idx_u)
            setattr(self, 'idx_r' + str(i), idx_r)
            setattr(self, 'idx_l' + str(i), idx_l)

    def forward(self, inputs):
        """
        :param inputs: tuple((B, C2, H2, W2),  (B, C3, H3, W3), ...)
        :return:
        """
        x = inputs[-1]      # (B, C_in, H, W)
        x = self.channel_reducer(x).clone()

        for direction in self.directions:
            for i in range(self.num_iters):
                conv = getattr(self, 'conv_' + direction + str(i))
                idx = getattr(self, 'idx_' + direction + str(i))
                if direction in ['d', 'u']:
                    x.add_(self.alpha * F.relu(conv(x[..., idx, :])))
                else:
                    x.add_(self.alpha * F.relu(conv(x[..., idx])))
        return x
