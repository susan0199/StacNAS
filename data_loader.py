# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torchvision.datasets as torch_datasets
import torchvision.transforms as transforms

from auto_augment import CIFAR10Policy


def load_dataset(dataset, data_dir, cutout_length, validation, auto_aug):
    dataset = dataset.lower()
    if dataset == "cifar10":
        data_cls = torch_datasets.CIFAR10
        num_classes = 10
    elif dataset == "cifar100":
        data_cls = torch_datasets.CIFAR100
        num_classes = 100
    elif dataset == "mnist":
        data_cls = torch_datasets.MNIST
        num_classes = 10
    elif dataset == "fashionmnist":
        data_cls = torch_datasets.FashionMNIST
        num_classes = 10
    else:
        raise ValueError("unexpected dataset: {}".format(dataset))

    train_transform, valid_transform = \
        data_transforms(dataset, cutout_length, auto_aug)
    train_data = data_cls(root=data_dir,
                          train=True,
                          download=True,
                          transform=train_transform)

    shape = train_data.data.shape
    input_channels = 3 if len(shape) == 4 else 1
    assert shape[1] == shape[2], "unexpected shape: {}".format(shape)
    input_size = shape[1]

    res = [input_size, input_channels, num_classes, train_data]

    if validation:
        valid_data = data_cls(root=data_dir,
                              train=False,
                              download=True,
                              transform=valid_transform)
        res.append(valid_data)

    return res


def data_transforms(dataset, cutout_length, auto_aug):
    dataset = dataset.lower()
    if dataset == "cifar10":
        MEAN = [0.49139968, 0.48215827, 0.44653124]
        STD = [0.24703233, 0.24348505, 0.26158768]
        random_transform = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()]
        if auto_aug:
            random_transform.append(CIFAR10Policy())
    elif dataset == "cifar100":
        MEAN = [0.50707519, 0.48654887, 0.44091785]
        STD = [0.26733428, 0.25643846, 0.27615049]
        random_transform = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()]
        if auto_aug:
            random_transform.append(CIFAR10Policy())
    elif dataset == "mnist":
        MEAN = [0.13066051707548254]
        STD = [0.30810780244715075]
        random_transform = [
            transforms.RandomAffine(degrees=15,
                                    translate=(0.1, 0.1),
                                    scale=(0.9, 1.1),
                                    shear=0.1)]
    elif dataset == "fashionmnist":
        MEAN = [0.28604063146254594]
        STD = [0.35302426207299326]
        random_transform = [
            transforms.RandomAffine(degrees=15,
                                    translate=(0.1, 0.1),
                                    scale=(0.9, 1.1),
                                    shear=0.1),
            transforms.RandomVerticalFlip()]
    else:
        raise ValueError("unexpected dataset: {}".format(dataset))

    normalize_transform = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)]

    if cutout_length > 0:
        cutout_transform = [Cutout(cutout_length)]
    else:
        cutout_transform = []

    train_transform = transforms.Compose(
        random_transform + normalize_transform + cutout_transform)
    valid_transform = transforms.Compose(normalize_transform)
    return train_transform, valid_transform


class Cutout(object):

    def __init__(self, length):
        self._length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self._length // 2, 0, h)
        y2 = np.clip(y + self._length // 2, 0, h)
        x1 = np.clip(x - self._length // 2, 0, w)
        x2 = np.clip(x + self._length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img
