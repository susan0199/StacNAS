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

from config import SearchConfig
from data_loader import load_dataset
import genotypes as gts
from search_cnn import SearchCNN
import utils


config = SearchConfig()
config.alpha_dir = os.path.join(config.stage_dir, "alphas")
os.system("mkdir -p {}".format(config.alpha_dir))

device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=config.log_dir)
writer.add_text("config", config.as_markdown(), 0)

logger = utils.get_logger(
    os.path.join(config.log_dir, "{}_{}.log".format(
        config.name, config.stage)))
config.print_args(logger.info)


def train(data_loader,
          model,
          criterion,
          alpha_optim,
          weight_optim,
          lr,
          epoch):
    loss = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    global_step = epoch * len(data_loader)
    writer.add_scalar("train/lr", lr, global_step)

    model.train()

    for step, (images, labels) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        num_samples = images.size(0)

        alpha_optim.zero_grad()
        weight_optim.zero_grad()

        logits, aux_logits = model(images)
        losses = criterion(logits, labels)
        if config.aux_weight > 0:
            losses += config.aux_weight * criterion(aux_logits, labels)
        losses.backward()

        nn.utils.clip_grad_norm_(model.weights(), config.grad_clip)
        if config.alpha_share or global_step >= 0:
            alpha_optim.step()
        weight_optim.step()

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
            
            prec_1, prec_5 = utils.accuracy(logits, labels, topk=(1, 5))
            loss.update(losses.item(), num_samples)
            top1.update(prec_1.item(), num_samples)
            top5.update(prec_5.item(), num_samples)

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


def parse_primitives():
    if config.stage == "search1":
        #ops_list = gts.OPS_LIST[:]
        ops_list = [ops_tuple[0] for ops_tuple in gts.OPS_DICT]
        num_alphas = 2 if config.alpha_share else config.num_cells
        primitives = [
            gts.build_primitive_from_init(config.num_nodes, ops_list) 
            for _ in range(num_alphas)]
    elif config.stage == "search2":
        alpha_file = os.path.join(
            config.alpha_dir.replace("search2", "search1"), "alphas_best.pk")
        with open(alpha_file, "rb") as f:
            alphas = pickle.load(f)
        primitives = [
            gts.build_primitive_from_alpha(alpha, gts.OPS_DICT) 
            for alpha in alphas]
    else:
        raise ValueError("unexpected stage: {}".format(config.stage))
    return primitives


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
    input_size, channels_in, num_classes, train_data = load_dataset(
        dataset=config.dataset,
        data_dir=config.data_dir,
        cutout_length=0,
        validation=False,
        auto_aug=config.auto_aug)

    num_samples = len(train_data)
    num_trains = int(config.train_ratio * num_samples)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=config.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(
            list(range(num_trains))),
        num_workers=config.num_workers,
        pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=config.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(
            list(range(num_trains, num_samples))),
        num_workers=config.num_workers,
        pin_memory=True)

    logger.info("parsing primitives...")
    primitives = parse_primitives()

    logger.info("building model...")
    model = SearchCNN(input_size=input_size,
                      channels_in=channels_in,
                      channels_init=config.init_channels,
                      num_cells=config.num_cells,
                      num_nodes=config.num_nodes,
                      num_classes=num_classes, 
                      stem_multiplier=3,
                      auxiliary=(config.aux_weight > 0),
                      primitives=primitives,
                      alpha_share=config.alpha_share)
    model = model.to(device)

    # logger.info("loading model...")
    # model = utils.load_checkpoint(config.model_dir)
    # model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    alpha_optim = torch.optim.Adam(params=model.alphas(),
                                   lr=config.alpha_learning_rate,
                                   betas=(0.5, 0.999),
                                   weight_decay=config.alpha_weight_decay)

    weight_optim = torch.optim.SGD(params=model.weights(),
                                   lr=config.learning_rate,
                                   momentum=config.momentum,
                                   weight_decay=config.weight_decay)

    if config.power_lr:
        lr_scheduler = utils.CosinePowerAnnealingLR(
            optimizer=weight_optim,
            T_max=config.epochs,
            eta_min=config.learning_rate_min,
            p=2)
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=weight_optim,
            T_max=config.epochs,
            eta_min=config.learning_rate_min)

    logger.info("start training...")
    history_top1 = []
    best_top1 = 0.0

    for epoch in range(config.epochs):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]
        logger.info("epoch: {:d}, lr: {:e}".format(epoch, lr))

        model.print_alphas(logger)

        train(train_loader, model, criterion, alpha_optim, weight_optim, 
              lr, epoch)

        if config.train_ratio < 1:
            global_step = (epoch + 1) * len(train_loader) - 1
            valid_top1 = valid(
                valid_loader, model, criterion, epoch, global_step)
            history_top1.append(valid_top1)

            if epoch == 0 or best_top1 < valid_top1:
                best_top1 = valid_top1
                is_best = True
            else:
                is_best = False
        else:
            is_best = True

        model.save_alphas(config.alpha_dir, is_best=is_best, logger=logger)
        utils.save_checkpoint(model, config.model_dir, is_best=is_best)

    with open(os.path.join(config.stage_dir, "history_top1.pk"), "wb") as f:
        pickle.dump(history_top1, f)

    logger.info("Final best valid Prec@1: {:.4%}".format(best_top1))
    logger.info("*** Finish {} ***".format(config.stage))


if __name__ == "__main__":
    main()
