# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn

from config import SearchConfig
from data_loader import load_dataset
import utils


config = SearchConfig()
config.model_dir = os.path.join(config.save_dir, "augment/models")

device = torch.device("cuda")

logger = utils.get_logger(
    os.path.join(config.log_dir, "{}_{}.log".format(
        config.name, config.stage)))


def test(data_loader, model, criterion):
    loss = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (images, labels) in enumerate(data_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            num_samples = images.size(0)

            logits, _ = model(images)
            losses = criterion(logits, labels)
            
            prec1, prec5 = utils.accuracy(logits, labels, topk=(1, 5))
            loss.update(losses.item(), num_samples)
            top1.update(prec1.item(), num_samples)
            top5.update(prec5.item(), num_samples)

            if step % config.report_freq == 0 or step == len(data_loader) - 1:
                logger.info("Test, Step: [{:3d}/{}], " \
                            "Loss: {:.4f}, Prec@(1,5): {:.4%}, {:.4%}".format(
                                step, len(data_loader), 
                                loss.avg, top1.avg, top5.avg))

    logger.info("Test, Final Prec@1: {:.4%}".format(top1.avg))
    return top1.avg


def main():
    if not torch.cuda.is_available():
        logger.info("no gpu device available")
        sys.exit(1)

    logger.info("*** Begin {} ***".format(config.stage))

    # set default gpu device
    torch.cuda.set_device(config.gpus[0])

    # set random seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    # get data with meta info
    logger.info("preparing data...")
    input_size, channels_in, num_classes, train_data, valid_data = \
        load_dataset(dataset=config.dataset,
                     data_dir=config.data_dir,
                     cutout_length=config.cutout_length,
                     validation=True,
                     auto_aug=config.auto_aug)

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_data,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True)

    logger.info("loading model...")
    if config.load_model_dir is not None:
        model = torch.load(config.load_model_dir)
    else:
        model = utils.load_checkpoint(config.model_dir)
    model = model.to(device)

    model_size = utils.param_size(model)
    logger.info("model_size: {:.3f} MB".format(model_size))

    if config.label_smooth > 0:
        criterion = utils.CrossEntropyLabelSmooth(
            num_classes, config.label_smooth)
    else:
        criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    logger.info("start testing...")
    best_top1 = test(valid_loader, model, criterion)

    logger.info("Final Prec@1: {:.4%}".format(best_top1))
    logger.info("*** Finish {} ***".format(config.stage))


if __name__ == "__main__":
    main()
