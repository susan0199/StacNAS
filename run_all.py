# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time

name = "stacnas"
dataset = "cifar10"

data_dir = "/home/user/datasets/"
save_dir = "/home/user/results/"

seed = 1
# seed = int(time.time()) % 1000

batch_size_search = 64
batch_size_augment = 96
num_cells_search1 = 14
num_cells_search2 = 20
num_cells_augment = 20
aux_weight_search = 0.0
aux_weight_augment = 0.4
epochs_search = 80
epochs_augment = 600

alpha_share = True
power_lr = False
auto_aug = False
label_smooth = 0.0
prefix = "g4"

extra = ""
if alpha_share:
    extra += " --alpha_share"
if power_lr:
    extra += " --power_lr"
if auto_aug:
    extra += " --auto_aug"

name_str = "_{}_{}_cell_{}_{}_{}_bs_{}_{}_epoch_{}_{}" \
    "_alpha_share_{}_aux_{}_{}_seed_{}".format(
        dataset, prefix, num_cells_search1, num_cells_search2, 
        num_cells_augment, batch_size_search, batch_size_augment, 
        epochs_search, epochs_augment, alpha_share, 
        aux_weight_search, aux_weight_augment, seed)

save_dir = save_dir[:-1] + name_str
save_folder = save_dir.split("/")[-1]

run_string_search1 = \
    "python search.py " \
    + "--name {} ".format(name) \
    + "--dataset {} ".format(dataset) \
    + "--train_ratio {} ".format(1.0) \
    + "--data_dir {} ".format(data_dir) \
    + "--save_dir {} ".format(save_dir) \
    + "--seed {} ".format(seed) \
    + "--stage {} ".format("search1") \
    + "--batch_size {} ".format(batch_size_search) \
    + "--init_channels {} ".format(16) \
    + "--num_cells {} ".format(num_cells_search1) \
    + "--grad_clip {} ".format(3) \
    + "--aux_weight {} ".format(aux_weight_search) \
    + "--epochs {} ".format(epochs_search) \
    + extra

run_string_search2 = \
    "python search.py " \
    + "--name {} ".format(name) \
    + "--dataset {} ".format(dataset) \
    + "--train_ratio {} ".format(1.0) \
    + "--data_dir {} ".format(data_dir) \
    + "--save_dir {} ".format(save_dir) \
    + "--seed {} ".format(seed) \
    + "--stage {} ".format("search2") \
    + "--batch_size {} ".format(batch_size_search) \
    + "--init_channels {} ".format(16) \
    + "--num_cells {} ".format(num_cells_search2) \
    + "--grad_clip {} ".format(3) \
    + "--aux_weight {} ".format(aux_weight_search) \
    + "--epochs {} ".format(epochs_search) \
    + extra

run_string_augment = \
    "python augment.py " \
    + "--name {} ".format(name) \
    + "--dataset {} ".format(dataset) \
    + "--data_dir {} ".format(data_dir) \
    + "--save_dir {} ".format(save_dir) \
    + "--seed {} ".format(3) \
    + "--stage {} ".format("augment") \
    + "--batch_size {} ".format(batch_size_augment) \
    + "--init_channels {} ".format(36) \
    + "--num_cells {} ".format(num_cells_augment) \
    + "--grad_clip {} ".format(5) \
    + "--aux_weight {} ".format(aux_weight_augment) \
    + "--label_smooth {} ".format(label_smooth) \
    + "--epochs {} ".format(epochs_augment) \
    + extra

run_string_feature = \
    "python feature.py " \
    + "--name {} ".format(name) \
    + "--dataset {} ".format(dataset) \
    + "--data_dir {} ".format(data_dir) \
    + "--save_dir {} ".format(save_dir) \
    + "--seed {} ".format(3) \
    + "--stage {} ".format("search1") \
    + "--batch_size {} ".format(batch_size_search)

run_string_test = \
    "python test.py " \
    + "--name {} ".format(name) \
    + "--dataset {} ".format(dataset) \
    + "--data_dir {} ".format(data_dir) \
    + "--save_dir {} ".format(save_dir) \
    + "--seed {} ".format(3) \
    + "--stage {} ".format("test") \
    + "--batch_size {} ".format(batch_size_augment)


os.system(run_string_search1)

os.system(run_string_search2)

os.system(run_string_augment)

# os.system(run_string_feature)

# os.system(run_string_test)
