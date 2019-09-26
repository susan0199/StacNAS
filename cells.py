# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import operators as ops


class SearchCell(nn.Module):
    """Search cell.

    Arguments:
        channels_prev_prev: output channels of cell_{k-2}.
        channels_prev: output channels of cell_{k-1}.
        channels: current channels of cell_{k}.
        reduction_prev: whether the previous cell is reduction.
        reduction: whether the current cell is reduction.
        genotype: a list of lists of (op_name, edge_id) tuples.
        primitive: a list of lists of operator names.
    """

    def __init__(self,
                 channels_prev_prev,
                 channels_prev,
                 channels,
                 reduction_prev,
                 reduction,
                 genotype=None,
                 primitive=None,
                 alpha_share=True,
                 **kwargs):
        super(SearchCell, self).__init__()
        self._cell_id = kwargs.get("cell_id", 0)
        self._channels_prev_prev = channels_prev_prev
        self._channels_prev = channels_prev
        self._channels = channels
        self._reduction_prev = reduction_prev
        self._reduction = reduction
        self._genotype = genotype
        self._primitive = primitive
        self._prepare()
        self._build_dag()
        if not alpha_share:
            self._build_alpha()

    def _prepare(self, affine=False):
        # If cell_{k-1} is reduction, output_{k-2} should 
        # be reduced to match the current input size.
        if self._reduction_prev:
            self._preproc0 = ops.FactorizedReduce(
                self._channels_prev_prev, self._channels, affine)
        else:
            self._preproc0 = ops.StdConv(
                self._channels_prev_prev, self._channels, 1, 1, 0, affine)
        self._preproc1 = ops.StdConv(
            self._channels_prev, self._channels, 1, 1, 0, affine)

    def _build_dag(self):
        # create dag from primitive
        self._dag = nn.ModuleList()
        for node_id in range(len(self._primitive)):
            self._dag.append(nn.ModuleList())
            for edge_id in range(node_id + 2):
                # reduction is only for input nodes
                stride = 2 if self._reduction and edge_id < 2 else 1
                op = ops.MixedOp(channels=self._channels,
                                 stride=stride,
                                 op_names=self._primitive[node_id][edge_id],
                                 cell_id=self._cell_id,
                                 node_id=node_id,
                                 edge_id=edge_id)
                self._dag[node_id].append(op)

    def _build_alpha(self):
        self._alpha = []
        for node_id in range(len(self._primitive)):
            self._alpha.append([])
            for edge_id in range(node_id + 2):
                num_ops = len(self._primitive[node_id][edge_id])
                new_alpha = nn.Parameter(torch.randn(num_ops) * 1e-3)
                name = "_alpha_{}_{}".format(node_id, edge_id)
                setattr(self, name, new_alpha)
                self._alpha[node_id].append(new_alpha)

    def forward(self, s0, s1, alpha=None, **kwargs):
        states = [self._preproc0(s0), self._preproc1(s1)]
        if alpha is None:
            alpha = [
                [F.softmax(edge_alpha, dim=-1) for edge_alpha in node_alpha] 
                for node_alpha in self._alpha]
        for edge_ops, edge_alpha in zip(self._dag, alpha):
            sk = sum([
                op(s, a, **kwargs) 
                for op, s, a in zip(edge_ops, states, edge_alpha)])
            states.append(sk)

        return torch.cat(states[2:], dim=1)


class AugmentCell(SearchCell):

    def _prepare(self, affine=True):
        super(AugmentCell, self)._prepare(affine)

    def _build_dag(self):
        # create dag from genotype
        self._dag = nn.ModuleList()
        for node_id, edges in enumerate(self._genotype):
            self._dag.append(nn.ModuleList())
            for op_name, edge_id in edges:
                # reduction is only for input nodes
                stride = 2 if self._reduction and edge_id < 2 else 1
                op = ops.OPS.get(op_name, "none")(self._channels, stride, True)
                if not isinstance(op, ops.Identity):
                    op = nn.Sequential(op, ops.DropPath())
                op.edge_id = edge_id
                self._dag[node_id].append(op)

    def forward(self, s0, s1, **kwargs):
        states = [self._preproc0(s0), self._preproc1(s1)]
        for edge_ops in self._dag:
            sk = sum([op(states[op.edge_id]) for op in edge_ops])
            states.append(sk)

        return torch.cat(states[2:], dim=1)
