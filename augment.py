# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import pickle
import sys
from tensorboardX import SummaryWriter
import time
import torch
import torch.nn as nn

from augment_cnn import AugmentCNN
from config import SearchConfig
from data_loader import load_dataset
import utils


config = SearchConfig()
config.alpha_dir = os.path.join(config.save_dir, "search2/alphas")

device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=config.log_dir)
writer.add_text("config", config.as_markdown(), 0)

logger = utils.get_logger(
    os.path.join(config.log_dir, "{}_{}.log".format(
        config.name, config.stage)))
config.print_args(logger.info)


def train(data_loader, model, criterion, optimizer, epoch):
    loss = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    lr = optimizer.param_groups[0]["lr"]
    global_step = epoch * len(data_loader)
    writer.add_scalar("train/lr", lr, global_step)

    model.train()

    for step, (images, labels) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        num_samples = images.size(0)

        optimizer.zero_grad()

        logits, aux_logits = model(images)
        losses = criterion(logits, labels)
        if config.aux_weight > 0:
            losses += config.aux_weight * criterion(aux_logits, labels)
        losses.backward()

        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        
        prec1, prec5 = utils.accuracy(logits, labels, topk=(1, 5))
        loss.update(losses.item(), num_samples)
        top1.update(prec1.item(), num_samples)
        top5.update(prec5.item(), num_samples)

        if step % config.report_freq == 0 or step == len(data_loader) - 1:
            logger.info("Train, Epoch: [{:3d}/{}], Step: [{:3d}/{}], " \
                        "Loss: {:.4f}, Prec@(1,5): {:.4%}, {:.4%}".format(
                            epoch, config.epochs, step, len(data_loader), 
                            loss.avg, top1.avg, top5.avg))

        writer.add_scalar("train/loss", losses.item(), global_step)
        writer.add_scalar("train/top1", prec1.item(), global_step)
        writer.add_scalar("train/top5", prec5.item(), global_step)
        global_step += 1

    logger.info("Train, Epoch: [{:3d}/{}], Final Prec@1: {:.4%}".format(
                epoch, config.epochs, top1.avg))


def valid(data_loader, model, criterion, epoch, global_step):
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
                logger.info("Valid, Epoch: [{:3d}/{}], Step: [{:3d}/{}], " \
                            "Loss: {:.4f}, Prec@(1,5): {:.4%}, {:.4%}".format(
                                epoch, config.epochs, step, len(data_loader), 
                                loss.avg, top1.avg, top5.avg))

    writer.add_scalar("valid/loss", loss.avg, global_step)
    writer.add_scalar("valid/top1", top1.avg, global_step)
    writer.add_scalar("valid/top5", top5.avg, global_step)

    logger.info("Valid, Epoch: [{:3d}/{}], Final Prec@1: {:.4%}".format(
                epoch, config.epochs, top1.avg))

    return top1.avg


def parse_genotypes():
    genotype_file = os.path.join(config.alpha_dir, "genotypes_best.pk")
    with open(genotype_file, "rb") as f:
        genotypes = pickle.load(f)
    return genotypes


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

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_data,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True)

    logger.info("parsing genotypes...")
    genotypes = parse_genotypes()
    logger.info(genotypes)
    
    logger.info("building model...")
    model = AugmentCNN(input_size=input_size,
                       channels_in=channels_in,
                       channels_init=config.init_channels,
                       num_cells=config.num_cells,
                       num_nodes=config.num_nodes,
                       num_classes=num_classes, 
                       stem_multiplier=3,
                       auxiliary=(config.aux_weight > 0),
                       genotypes=genotypes,
                       alpha_share=config.alpha_share)
    model = model.to(device)

    model_size = utils.param_size(model)
    logger.info("model_size: {:.3f} MB".format(model_size))

    if config.label_smooth > 0:
        criterion = utils.CrossEntropyLabelSmooth(
            num_classes, config.label_smooth)
    else:
        criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=config.learning_rate,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    if config.power_lr:
        lr_scheduler = utils.CosinePowerAnnealingLR(
            optimizer=optimizer, T_max=config.epochs, p=2)
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=config.epochs)

    logger.info("start training...")
    history_top1 = []
    best_top1 = 0.0

    for epoch in range(config.epochs):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]
        logger.info("epoch: {:d}, lr: {:e}".format(epoch, lr))

        drop_prob = config.drop_path_prob * epoch / config.epochs
        model.drop_path_prob(drop_prob)

        train(train_loader, model, criterion, optimizer, epoch)

        global_step = (epoch + 1) * len(train_loader) - 1
        valid_top1 = valid(valid_loader, model, criterion, epoch, global_step)
        history_top1.append(valid_top1)

        if epoch == 0 or best_top1 < valid_top1:
            best_top1 = valid_top1
            is_best = True
        else:
            is_best = False

        utils.save_checkpoint(model, config.model_dir, is_best=is_best)

    with open(os.path.join(config.stage_dir, "history_top1.pk"), "wb") as f:
        pickle.dump(history_top1, f)

    logger.info("Final best valid Prec@1: {:.4%}".format(best_top1))
    logger.info("*** Finish {} ***".format(config.stage))


if __name__ == "__main__":
    main()
