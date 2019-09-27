# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from functools import partial
import os
import torch


def get_parser(name):
    parser = argparse.ArgumentParser(
        prog=name,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # always print default values
    parser.add_argument = partial(parser.add_argument, help="")
    return parser


def parse_gpu(gpu):
    if gpu == "all":
        return list(range(torch.cuda.device_count()))
    else:
        return [int(x.strip()) for x in gpu.split(",")]


class BaseConfig(argparse.Namespace):

    def as_markdown(self):
        text = "|name|value|\n|-|-|\n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}| \n".format(attr, value)
        return text

    def print_args(self, method=print):
        method("Arguments:")
        method("-" * 20)
        for attr, value in sorted(vars(self).items()):
            method("{}={}".format(attr.upper(), value))
        method("-" * 20)


class SearchConfig(BaseConfig):

    def __init__(self):
        parser = get_parser("search_config")
        parser.add_argument(
            "--name", required=True, help="job name")
        parser.add_argument(
            "--dataset", type=str, default="CIFAR10", 
            help="CIFAR10/CIFAR100/MNIST/FashionMNIST")
        parser.add_argument(
            "--data_dir", type=str, 
            default="/home/user/datasets/", 
            help="path to load data")
        parser.add_argument(
            "--save_dir", type=str, 
            default="/home/user/results/", 
            help="path to save results")
        parser.add_argument(
            "--load_model_dir", type=str, 
            default=None, 
            help="path to load models, only for testing")

        parser.add_argument(
            "--seed", type=int, default=1, help="random seed")
        parser.add_argument(
            "--gpu", type=str, default="0", 
            help="gpu device ids separated by commas")
        parser.add_argument(
            "--num_workers", type=int, default=4, 
            help="number of workers for data loader")
        parser.add_argument(
            "--stage", type=str, default="search1", 
            help="search1/search2/augment")
        parser.add_argument(
            "--train_ratio", type=float, default=1, 
            help="train-valid split ratio")
        
        parser.add_argument(
            "--batch_size", type=int, default=64, help="batch size")
        parser.add_argument(
            "--init_channels", type=int, default=16, 
            help="number of initial channels")
        parser.add_argument(
            "--num_cells", type=int, default=8, help="number of cells")
        parser.add_argument(
            "--num_nodes", type=int, default=4, 
            help="number of intermediate nodes in a cell")
        
        parser.add_argument(
            "--power_lr", action="store_true", default=False, 
            help="whether to use cosine power annealing lr")
        parser.add_argument(
            "--learning_rate", type=float, default=0.025, 
            help="initial learning rate")
        parser.add_argument(
            "--learning_rate_min", type=float, default=1e-3, 
            help="min learning rate")
        parser.add_argument(
            "--momentum", type=float, default=0.9, help="momentum")
        parser.add_argument(
            "--weight_decay", type=float, default=3e-4, help="weight decay")
        parser.add_argument(
            "--grad_clip", type=float, default=3, help="gradient clip")

        parser.add_argument(
            "--alpha_share", action="store_true", default=False, 
            help="whether to use shared alpha_normal/reduce, " \
                 "or independent alphas for all cells")
        parser.add_argument(
            "--alpha_learning_rate", type=float, default=3e-4, 
            help="learning rate for architecture encoding")
        parser.add_argument(
            "--alpha_weight_decay", type=float, default=1e-3, 
            help="weight decay for architecture encoding")

        parser.add_argument(
            "--auto_aug", action="store_true", default=False, 
            help="whether to use auto augmentation of images")
        parser.add_argument(
            "--aux_weight", type=float, default=0, 
            help="auxiliary loss weight, " \
                 "aux_weight=0 for no auxiliary head")
        parser.add_argument(
            "--cutout_length", type=int, default=16, help="cutout length")
        parser.add_argument(
            "--drop_path_prob", type=float, default=0.2, 
            help="drop path probability")
        parser.add_argument(
            "--label_smooth", type=float, default=0.0, 
            help="label smoothing, label_smooth=0 for no label smoothing")     
        parser.add_argument(
            "--epochs", type=int, default=80, help="number of epochs")
        parser.add_argument(
            "--report_freq", type=float, default=50, help="report frequency")
        
        args, unparsed = parser.parse_known_args()
        super(SearchConfig, self).__init__(**vars(args))
        self.gpus = parse_gpu(self.gpu)
        self.stage_dir = os.path.join(self.save_dir, self.stage)
        self.log_dir = os.path.join(self.stage_dir, "logs")
        os.system("mkdir -p {}".format(self.log_dir))
        self.model_dir = os.path.join(self.stage_dir, "models")
        os.system("mkdir -p {}".format(self.model_dir))


if __name__ == "__main__":
    config = SearchConfig()
    text = config.as_markdown()
    print(text)
    config.print_args()
