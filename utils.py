# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import numpy as np
import os
import shutil
import torch
import torch.nn as nn


def accuracy(output, target, topk=(1,)):
    """Compute precision@k for the specified values of k."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


class AverageMeter(object):
    """Calculate averages as process progresses."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cnt = 0
        self.avg = 0
        self.sum = 0

    def update(self, val, n=1):
        self.cnt += n
        self.sum += val * n
        self.avg = self.sum / self.cnt


class CosinePowerAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    """Cosine power annealing from: https://arxiv.org/abs/1903.09900."""

    def __init__(self, optimizer, T_max, eta_min=0, p=2, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.p = p
        super(CosinePowerAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        tmp = (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
        if self.p != 1:
            tmp = (math.pow(self.p, tmp + 1) - self.p) / (self.p ** 2 - self.p)

        return [self.eta_min + (base_lr - self.eta_min) * tmp 
                for base_lr in self.base_lrs]


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, labels):
        log_probs = self.log_softmax(inputs)
        labels = \
            torch.zeros_like(log_probs).scatter_(1, labels.unsqueeze(1), 1)
        labels = (1 - self.epsilon) * labels + self.epsilon / self.num_classes
        loss = - (labels * log_probs).mean(0).sum()
        return loss


def get_logger(log_file):
    logger = logging.getLogger()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(message)s",
        datefmt="%m/%d %I:%M:%S %p",
        style="%")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)
    return logger


def param_size(model):
    """Count parameter size in MB."""
    num_params = np.sum(
        np.prod(v.size()) for k, v in model.named_parameters() 
        if not k.startswith("_aux_head"))
    return num_params / 1024. / 1024.


def save_checkpoint(model, save_dir, epoch=None, is_best=False):
    if epoch is not None:
        ckpt = os.path.join(save_dir, "checkpoint_{}.pth.tar".format(epoch))
    else:
        ckpt = os.path.join(save_dir, "checkpoint.pth.tar")
    torch.save(model, ckpt)
    if is_best:
        best_ckpt = os.path.join(save_dir, "best.pth.tar")
        shutil.copyfile(ckpt, best_ckpt)


def load_checkpoint(load_dir, epoch=None, is_best=True):
    if is_best:
        ckpt = os.path.join(load_dir, "best.pth.tar")
    elif epoch is not None:
        ckpt = os.path.join(load_dir, "checkpoint_{}.pth.tar".format(epoch))
    else:
        ckpt = os.path.join(load_dir, "checkpoint.pth.tar")
    model = torch.load(ckpt)
    return model
