# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import pickle
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F

from augment_cnn import AuxiliaryHead
from cells import SearchCell
import genotypes as gts


class SearchCNN(nn.Module):
    """Search CNN model.

    Arguments:
        input_size: image height/width.
        channels_in: input image channels.
        channels_init: initial cell channels.
        num_cells: number of cells.
        num_nodes: number of intermediate nodes in a cell.
        num_classes: number of classes.
        stem_multiplier:
        auxiliary: whether to add the auxiliary head.
        primitives:
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
                 primitives,
                 alpha_share=True,
                 **kwargs):
        super(SearchCNN, self).__init__()
        self._input_size = input_size
        self._channels_in = channels_in
        self._channels_init = channels_init
        self._num_cells = num_cells
        self._num_nodes = num_nodes
        self._num_classes = num_classes
        self._stem_multiplier = stem_multiplier
        self._auxiliary = auxiliary
        self._aux_pos = 2 * self._num_cells // 3 if self._auxiliary else -1
        self._primitives = primitives
        self._alpha_share = alpha_share
        if self._alpha_share:
            self._alpha_types = ["normal", "reduce"]
        else:
            self._alpha_types = \
                ["cell_{}".format(i) for i in range(self._num_cells)]

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
                    primitive = self._primitives[1]
                else:
                    primitive = self._primitives[cell_id]
            else:
                reduction = False
                if self._alpha_share:
                    primitive = self._primitives[0]
                else:
                    primitive = self._primitives[cell_id]

            cell = SearchCell(channels_prev_prev=channels_prev_prev,
                              channels_prev=channels_prev,
                              channels=channels,
                              reduction_prev=reduction_prev,
                              reduction=reduction,
                              primitive=primitive,
                              alpha_share=self._alpha_share,
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

        self._build_alphas()

        self._param_alphas, self._param_weights = [], []
        for name, param in self.named_parameters():
            if "alpha" in name:
                self._param_alphas.append((name, param))
            else:
                self._param_weights.append((name, param))

    def _build_alpha_v1(self, num_ops, tp):
        # for edges with same number of operators
        alpha = nn.ParameterList([
            nn.Parameter(torch.randn(node_id + 2, num_ops) * 1e-3) 
            for node_id in range(self._num_nodes)])
        setattr(self, "_alpha_{}".format(tp), alpha)
        return alpha

    def _build_alpha_v2(self, primitive, tp):
        # for edges with different number of operators
        alpha = []
        for node_id in range(self._num_nodes):
            alpha.append([])
            for edge_id in range(node_id + 2):
                num_ops = len(primitive[node_id][edge_id])
                new_alpha = nn.Parameter(torch.randn(num_ops) * 1e-3)
                name = "_alpha_{}_{}_{}".format(tp, node_id, edge_id)
                setattr(self, name, new_alpha)
                alpha[node_id].append(new_alpha)
        return alpha

    def _build_alphas(self):
        if self._alpha_share:
            # self._alphas = [
            #     self._build_alpha_v1(len(primitive[0][0]), tp) 
            #     for primitive, tp in zip(self._primitives, self._alpha_types)]
            self._alphas = [
                self._build_alpha_v2(primitive, tp) 
                for primitive, tp in zip(self._primitives, self._alpha_types)]
        else:
            self._alphas = [cell._alpha for cell in self._cells]

    def forward(self, x, **kwargs):
        if self._alpha_share:
            softmax_alphas = [[
                [F.softmax(edge_alpha, dim=-1) for edge_alpha in node_alpha] 
                for node_alpha in alpha] for alpha in self._alphas]
        else:
            softmax_alphas = [None, None]

        s0 = s1 = self._stem(x)

        aux_out = None
        for cell_id, cell in enumerate(self._cells):
            softmax_alpha = softmax_alphas[int(cell._reduction)]
            s0, s1 = s1, cell(s0, s1, softmax_alpha, **kwargs)
            if cell_id == self._aux_pos and self.training:
                aux_out = self._aux_head(s1)

        out = self._global_pooling(s1)
        out = out.view(out.size(0), -1)
        out = self._classifier(out)
        return out, aux_out

    def named_weights(self):
        return [(name, param) for name, param in self._param_weights]

    def weights(self):
        return [param for name, param in self._param_weights]

    def named_alphas(self):
        return [(name, param) for name, param in self._param_alphas]

    def alphas(self):
        return [param for name, param in self._param_alphas]

    def return_alphas(self):
        return self._alphas

    def print_alphas(self, logger):
        origin_formatters = []
        for handler in logger.handlers:
            origin_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        for alpha, tp in zip(self._alphas, self._alpha_types):
            head_str = "-------- alpha_{} --------".format(tp)
            logger.info(head_str.upper())
            for node_alpha in alpha:
                #logger.info(F.softmax(node_alpha, dim=-1))
                for edge_alpha in node_alpha:
                    logger.info(F.softmax(edge_alpha, dim=-1))
            logger.info("")

        logger.info("-" * len(head_str))

        for handler, formatter in zip(logger.handlers, origin_formatters):
            handler.setFormatter(formatter)

    def save_alphas(self, save_dir, epoch=None, is_best=False, logger=None):
        genotypes = gts.save_alphas(
            self._alphas, self._primitives, save_dir, epoch, is_best)
        if logger is not None:
            logger.info("epoch: {}, genotypes: {}".format(epoch, genotypes))
