# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import torch
import torch.nn as nn

from feature_map import save_features


OPS = {
    "none": lambda channels, stride, affine: Zero(stride),
    "skip_connect": lambda channels, stride, affine: 
        Identity() if stride == 1 \
        else FactorizedReduce(channels, channels, affine),
    "avg_pool_3x3": lambda channels, stride, affine: 
        PoolBN("avg", channels, 3, stride, 1, affine),
    "max_pool_3x3": lambda channels, stride, affine: 
        PoolBN("max", channels, 3, stride, 1, affine),
    "sep_conv_3x3": lambda channels, stride, affine: 
        SepConv(channels, channels, 3, stride, 1, affine),
    "sep_conv_5x5": lambda channels, stride, affine: 
        SepConv(channels, channels, 5, stride, 2, affine),
    "sep_conv_7x7": lambda channels, stride, affine: 
        SepConv(channels, channels, 7, stride, 3, affine),
    "dil_conv_3x3": lambda channels, stride, affine: 
        DilConv(channels, channels, 3, stride, 2, 2, affine),
    "dil_conv_5x5": lambda channels, stride, affine: 
        DilConv(channels, channels, 5, stride, 4, 2, affine),
    "conv_7x1_1x7": lambda channels, stride, affine: 
        FacConv(channels, channels, 7, stride, 3, affine),
}


class Zero(nn.Module):

    def __init__(self, stride, **kwargs):
        super(Zero, self).__init__()
        self._stride = stride

    def forward(self, x, **kwargs):
        if self._stride == 1:
            return x * 0.0
        else:
            return x[:, :, ::self._stride, ::self._stride] * 0.0


class Identity(nn.Module):

    def __init__(self, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x, **kwargs):
        return x


class PoolBN(nn.Module):
    """AvgPool/MaxPool - BN"""

    def __init__(self,
                 pool_type,
                 channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 affine=True,
                 **kwargs):
        super(PoolBN, self).__init__()

        pool_type = pool_type.lower()
        if pool_type == "avg":
            self.pool = nn.AvgPool2d(
                kernel_size, stride, padding, count_include_pad=False)
        elif pool_type == "max":
            self.pool = nn.MaxPool2d(
                kernel_size, stride, padding)
        else:
            raise ValueError(
                "unexpected pool type: {}".format(pool_type))

        self.bn = nn.BatchNorm2d(channels, affine=affine)

    def forward(self, x, **kwargs):
        out = self.pool(x)
        out = self.bn(out)
        return out


class StdConv(nn.Module):
    """ReLU - Conv - BN"""

    def __init__(self,
                 channels_in,
                 channels_out,
                 kernel_size,
                 stride=1,
                 padding=0,
                 affine=True,
                 **kwargs):
        super(StdConv, self).__init__()
        self._ops = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels_in,
                      channels_out,
                      kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=False),
            nn.BatchNorm2d(channels_out, affine=affine))

    def forward(self, x, **kwargs):
        return self._ops(x)


class FacConv(nn.Module):
    """ReLU - Conv(Kx1) - Conv(1xK) - BN"""

    def __init__(self,
                 channels_in,
                 channels_out,
                 kernel_size,
                 stride=1,
                 padding=0,
                 affine=True,
                 **kwargs):
        super(FacConv, self).__init__()
        self._ops = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels_in,
                      channels_in,
                      (kernel_size, 1),
                      stride=stride,
                      padding=padding,
                      bias=False),
            nn.Conv2d(channels_in,
                      channels_out,
                      (1, kernel_size),
                      stride=stride,
                      padding=padding,
                      bias=False),
            nn.BatchNorm2d(channels_out, affine=affine))

    def forward(self, x, **kwargs):
        return self._ops(x)


class DilConv(nn.Module):
    """ReLU - (Dilated) Depthwise Separable - Pointwise - BN"""

    def __init__(self,
                 channels_in,
                 channels_out,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 affine=True,
                 **kwargs):
        super(DilConv, self).__init__()
        self._ops = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels_in,
                      channels_in,
                      kernel_size,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      groups=channels_in,
                      bias=False),
            nn.Conv2d(channels_in,
                      channels_out,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(channels_out, affine=affine))

    def forward(self, x, **kwargs):
        return self._ops(x)


class SepConv(nn.Module):
    """DilConv(dilation=1) - DilConv(dilation=1)"""

    def __init__(self,
                 channels_in,
                 channels_out,
                 kernel_size,
                 stride=1,
                 padding=0,
                 affine=True,
                 **kwargs):
        super(SepConv, self).__init__()
        self._ops = nn.Sequential(
            DilConv(channels_in,
                    channels_in,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=1,
                    affine=affine),
            DilConv(channels_in,
                    channels_out,
                    kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=1,
                    affine=affine))

    def forward(self, x, **kwargs):
        return self._ops(x)


class FactorizedReduce(nn.Module):
    """Reduce featuremap size by factorized pointwise (stride=2)."""

    def __init__(self,
                 channels_in,
                 channels_out,
                 affine=True,
                 **kwargs):
        super(FactorizedReduce, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(channels_in,
                               channels_out // 2,
                               kernel_size=1,
                               stride=2,
                               padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(channels_in,
                               channels_out // 2,
                               kernel_size=1,
                               stride=2,
                               padding=0,
                               bias=False)
        self.bn = nn.BatchNorm2d(channels_out, affine=affine)

    def forward(self, x, **kwargs):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class MixedOp(nn.Module):
    """Mixed operator on edges."""

    def __init__(self,
                 channels,
                 stride,
                 op_names,
                 **kwargs):
        super(MixedOp, self).__init__()
        self._cell_id = kwargs.get("cell_id", 0)
        self._node_id = kwargs.get("node_id", 0)
        self._edge_id = kwargs.get("edge_id", 0)
        self._ops = nn.ModuleList([
            OPS.get(op_name, "none")(channels, stride, False) 
            for op_name in op_names])
    
    def forward(self, x, weights, **kwargs):
        outs = [op(x) for op in self._ops]
        mixed_out = sum([w * out for w, out in zip(weights, outs)])
        
        # save features to file
        if kwargs.get("save", False):
            feature_dir = kwargs.get("feature_dir", "./")
            feature_str = "cell{}_node{}_edge{}.pk".format(
                self._cell_id, self._node_id, self._edge_id)
            mode = kwargs.get("mode", "wb")
            save_features(
                outs[:-1], os.path.join(feature_dir, feature_str), mode)
        
        return mixed_out


class DropPath(nn.Module):

    def __init__(self, p=0.0, **kwargs):
        super(DropPath, self).__init__()
        self._p = p

    def extra_repr(self):
        return "p={}, inplace".format(self._p)

    def forward(self, x, **kwargs):
        _drop_path(x, self._p, self.training)
        return x


def _drop_path(x, drop_prob, training):
    if training and drop_prob > 0:
        keep_prob = 1.0 - drop_prob
        # mask per data point
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob).mul_(mask)
    return x
