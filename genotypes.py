# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from graphviz import Digraph
import os
import pickle
import shutil
import torch


OPS_LIST = [
    "avg_pool_3x3",
    "max_pool_3x3",
    "skip_connect",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_5x5",
    "none"]


OPS_DICT = [
    ("max_pool_3x3", ["avg_pool_3x3", "max_pool_3x3", "none"]),
    ("skip_connect", ["skip_connect", "skip_connect", "none"]),
    ("sep_conv_3x3", ["sep_conv_3x3", "sep_conv_5x5", "none"]),
    ("dil_conv_3x3", ["dil_conv_3x3", "dil_conv_5x5", "none"]),
    ("none", ["none"])]


def build_primitive_from_init(num_nodes, ops_list):
    primitive = [
        [ops_list for edge_id in range(node_id + 2)] 
        for node_id in range(num_nodes)]
    return primitive


def build_primitive_from_alpha(alpha, ops_dict):
    primitive = []
    for node_alpha in alpha:
        node_primitive = []
        for edge_alpha in node_alpha:
            # ignore the last op "none"
            _, op_ids = torch.topk(edge_alpha[:-1], 1)
            node_primitive.append(ops_dict[op_ids.item()][1])
        primitive.append(node_primitive)

    return primitive


def build_genotype_from_alpha(alpha, primitive, k):
    genotype = []
    for node_id, node_alpha in enumerate(alpha):
        top_alphas, top_ops = [], []
        for edge_id, edge_alpha in enumerate(node_alpha):
            # ignore the last op "none"
            vals, op_ids = torch.topk(edge_alpha[:-1], 1)
            top_alphas.append(vals[0])
            top_ops.append(primitive[node_id][edge_id][op_ids.item()])
        
        _, topk_edge_ids = torch.topk(torch.cuda.FloatTensor(top_alphas), k)
        genotype.append([
            (top_ops[edge_id], edge_id.item()) for edge_id in topk_edge_ids])

    return genotype


def save_alphas(alphas, primitives, save_dir, epoch=None, is_best=False):
    if epoch is not None:
        alpha_file = os.path.join(save_dir, "alphas_{}.pk".format(epoch))
    else:
        alpha_file = os.path.join(save_dir, "alphas.pk")
    with open(alpha_file, "wb") as f:
        pickle.dump(alphas, f)

    genotypes = [
        build_genotype_from_alpha(alpha, primitive, 2) 
        for alpha, primitive in zip(alphas, primitives)]
    if epoch is not None:
        genotype_file = os.path.join(save_dir, "genotypes_{}.pk".format(epoch))
    else:
        genotype_file = os.path.join(save_dir, "genotypes.pk")
    with open(genotype_file, "wb") as f:
        pickle.dump(genotypes, f)

    if is_best:
        shutil.copyfile(
            alpha_file, os.path.join(save_dir, "alphas_best.pk"))
        shutil.copyfile(
            genotype_file, os.path.join(save_dir, "genotypes_best.pk"))

        for i, genotype in enumerate(genotypes):
            plot_cell(genotype=genotype,
                      name="cell_{}".format(i),
                      save_dir=os.path.join(save_dir, "dags"))

    return genotypes


def load_alphas(load_dir, epoch=None, is_best=True):
    if is_best:
        alpha_file = os.path.join(load_dir, "alphas_best.pk")
    elif epoch is not None:
        alpha_file = os.path.join(load_dir, "alphas_{}.pk".format(epoch))
    else:
        alpha_file = os.path.join(load_dir, "alphas.pk")
    with open(alpha_file, "rb") as f:
        alphas = pickle.load(f)
    return alphas


def plot_cell(genotype, name, save_dir):
    graph_attr = {
        "fontsize": "20",
        "fontname": "times"}
    node_attr = {
        "shape": "rect",
        "style": "filled",
        "align": "center",
        "height": "0.5",
        "width": "0.5",
        "penwidth": "2",
        "fontsize": "20",
        "fontname": "helvetica"}
    edge_attr = {
        "fontsize": "16",
        "fontcolor": "dodgerblue2",
        "fontname": "times"}
    g = Digraph(name=name,
                directory=save_dir,
                format="png",
                engine="dot",
                graph_attr=graph_attr,
                node_attr=node_attr,
                edge_attr=edge_attr)
    g.body.extend(["rankdir=LR"])

    # input nodes
    g.node("c_{k-2}", fillcolor="darkseagreen2")
    g.node("c_{k-1}", fillcolor="darkseagreen2")

    # intermediate nodes
    num_nodes = len(genotype)
    for node_id in range(num_nodes):
        g.node(str(node_id), fillcolor="lightblue")

    # output node
    g.node("c_{k}", fillcolor="palegoldenrod")

    # edges
    for node_id, edges in enumerate(genotype):
        v = str(node_id)
        for op_name, edge_id in edges:
            if edge_id == 0:
                u = "c_{k-2}"
            elif edge_id == 1:
                u = "c_{k-1}"
            else:
                u = str(edge_id - 2)
            g.edge(u, v, label=op_name, fillcolor="gray")
        g.edge(v, "c_{k}", fillcolor="gray")

    g.attr(label=name, overlap="false")
    g.render(view=False)
