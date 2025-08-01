# #!/usr/bin/python
from __future__ import print_function
#
# ### torch lib
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import autograd
import numpy as np
import os

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


class LastPart(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=40, kernel_size=3, num_cab=2, reduction=4, bias=False):
        super(LastPart, self).__init__()
        self.act = nn.PReLU()

        self.conv3 = conv(in_channels=3, out_channels=40, kernel_size=3, bias=bias)
        cab3 = [CAB(n_feat=n_feat, kernel_size=kernel_size, reduction=reduction, bias=bias, act=self.act) for _ in range(num_cab)]
        self.cab3 = nn.Sequential(*cab3)

        self.conv2 = conv(in_channels=6, out_channels=40, kernel_size=3, bias=bias)
        cab2 = [CAB(n_feat=n_feat, kernel_size=kernel_size, reduction=reduction, bias=bias, act=self.act) for _ in range(num_cab)]
        self.cab2 = nn.Sequential(*cab2)

        self.conv1 = conv(in_channels=9, out_channels=40, kernel_size=3, bias=bias)
        cab1 = [CAB(n_feat=n_feat, kernel_size=kernel_size, reduction=reduction, bias=bias, act=self.act) for _ in range(num_cab)]
        self.cab1 = nn.Sequential(*cab1)

        cab23 = [CAB(n_feat=n_feat, kernel_size=kernel_size, reduction=reduction, bias=bias, act=self.act) for _ in range(num_cab)]
        self.cab23 = nn.Sequential(*cab23)

        cab12 = [CAB(n_feat=n_feat, kernel_size=kernel_size, reduction=reduction, bias=bias, act=self.act) for _ in range(num_cab)]
        self.cab12 = nn.Sequential(*cab12)

        self.softmax23 = torch.nn.Softmax2d()
        self.softmax12 = torch.nn.Softmax2d()

        self.conv_last1 = conv(in_channels=40, out_channels=16, kernel_size=3, bias=bias)
        self.conv_last2 = conv(in_channels=16, out_channels=8, kernel_size=3, bias=bias)
        self.conv_last3 = conv(in_channels=8, out_channels=3, kernel_size=3, bias=bias)
        self.relu = nn.ReLU(inplace=True)

        self.conv_tail = conv(in_channels=3, out_channels=3, kernel_size=3, bias=bias)

    def forward(self, restored, rain):
        N = 3
        B, C, H , W = restored[0].size()

        stage3 = restored[0].cuda()
        stage2 = torch.cat([restored[0],restored[1]],dim=1).contiguous().cuda()
        stage1 = torch.cat([restored[0], restored[1], restored[2]], dim=1).contiguous().cuda()
        del(restored)

        stage3_feat = self.conv3(stage3)
        del(stage3)
        stage3_feat = self.cab3(stage3_feat)

        stage2_feat = self.conv2(stage2)
        del(stage2)
        stage2_feat = self.cab2(stage2_feat)

        stage23 = stage3_feat + stage2_feat
        stage23_feat = self.cab23(stage23)
        stage23_soft = self.softmax23(stage23_feat)
        del (stage23_feat)
        stage2_out = stage2_feat * stage23_soft
        del(stage23_soft)

        stage1_feat = self.conv1(stage1)
        del(stage1)
        stage1_feat = self.cab1(stage1_feat)

        stage12 = stage1_feat + stage2_feat
        del(stage2_feat)
        stage12_feat = self.cab12(stage12)
        del(stage12)
        stage12_soft = self.softmax12(stage12_feat)
        del(stage12_feat)
        stage1_out = stage1_feat * stage12_soft
        del(stage1_feat)
        del(stage12_soft)

        stage_sum = stage1_out + stage2_out + stage3_feat
        del(stage1_out)
        del(stage2_out)
        del(stage3_feat)
        last_feat = self.conv_last3(self.conv_last2(self.relu(self.conv_last1(stage_sum)))) + rain
        del(stage_sum)

        derain = self.conv_tail(last_feat)
        del(last_feat)
        del(rain)

        return derain
