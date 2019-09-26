# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

from cells import AugmentCell
import operators as ops


class AuxiliaryHead(nn.Module):
    """Auxiliary head.

    Add at 2/3 place of the network.
    """

    def __init__(self,
                 input_size,
                 channels,
                 num_classes,
                 **kwargs):
        super(AuxiliaryHead, self).__init__()
        assert input_size in [7, 8]
        self._features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=5,
                         stride=input_size - 5,
                         padding=0,
                         count_include_pad=False),
            nn.Conv2d(channels, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True))
        self._classifier = nn.Linear(768, num_classes)

    def forward(self, x, **kwargs):
        out = self._features(x)
        out = out.view(out.size(0), -1)
        out = self._classifier(out)
        return out


class AugmentCNN(nn.Module):
    """Augment CNN model.

    Arguments:
        input_size: image height/width.
        channels_in: input image channels.
        channels_init: initial cell channels.
        num_cells: number of cells.
        num_nodes: number of intermediate nodes in a cell.
        num_classes: number of classes.
        stem_multiplier:
        auxiliary: whether to add the auxiliary head.
        genotypes:
        alpha_share: whether to use shared alpha for cells.
    """

    def __init__(self,
                 input_size,
                 channels_in,
                 channels_init,
                 num_cells,
                 num_nodes,
                 num_classes,
                 stem_multiplier,
                 auxiliary,
                 genotypes,
                 alpha_share=True,
                 **kwargs):
        super(AugmentCNN, self).__init__()
        self._input_size = input_size
        self._channels_in = channels_in
        self._channels_init = channels_init
        self._num_cells = num_cells
        self._num_nodes = num_nodes
        self._num_classes = num_classes
        self._stem_multiplier = stem_multiplier
        self._auxiliary = auxiliary
        self._aux_pos = 2 * self._num_cells // 3 if self._auxiliary else -1
        self._genotypes = genotypes
        self._alpha_share = alpha_share

        # convolutional layers
        channels = self._channels_init * self._stem_multiplier
        self._stem = nn.Sequential(
            nn.Conv2d(self._channels_in,
                      channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(channels))

        self._cells = nn.ModuleList()
        channels_prev_prev, channels_prev, channels = \
            channels, channels, self._channels_init
        reduction_prev = False

        for cell_id in range(self._num_cells):
            if cell_id in [self._num_cells // 3, 2 * self._num_cells // 3]:
                reduction = True
                channels *= 2
                if self._alpha_share:
                    genotype = self._genotypes[1]
                else:
                    genotype = self._genotypes[cell_id]
            else:
                reduction = False
                if self._alpha_share:
                    genotype = self._genotypes[0]
                else:
                    genotype = self._genotypes[cell_id]

            cell = AugmentCell(channels_prev_prev=channels_prev_prev,
                               channels_prev=channels_prev,
                               channels=channels,
                               reduction_prev=reduction_prev,
                               reduction=reduction,
                               genotype=genotype,
                               cell_id=cell_id,
                               **kwargs)
            self._cells.append(cell)
            channels_prev_prev, channels_prev = \
                channels_prev, channels * self._num_nodes
            reduction_prev = reduction

            if cell_id == self._aux_pos:
                self._aux_head = AuxiliaryHead(
                    input_size=self._input_size // 4,
                    channels=channels_prev,
                    num_classes=self._num_classes,
                    **kwargs)

        self._global_pooling = nn.AdaptiveAvgPool2d(1)
        self._classifier = nn.Linear(channels_prev, self._num_classes)

    def forward(self, x, **kwargs):
        s0 = s1 = self._stem(x)

        aux_out = None
        for cell_id, cell in enumerate(self._cells):
            s0, s1 = s1, cell(s0, s1, **kwargs)
            if cell_id == self._aux_pos and self.training:
                aux_out = self._aux_head(s1)

        out = self._global_pooling(s1)
        out = out.view(out.size(0), -1)
        out = self._classifier(out)
        return out, aux_out

    def drop_path_prob(self, p):
        for module in self.modules():
            if isinstance(module, ops.DropPath):
                module._p = p
