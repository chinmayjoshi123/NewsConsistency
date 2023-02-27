# Original copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import math


def warmup_learning_rate(optimizer, curr_step, warmup_step, confs):
    """linearly warm up learning rate"""
    lr = confs["learning_rate"]
    scalar = float(curr_step) / float(max(1, warmup_step))
    scalar = min(1., max(0., scalar))
    lr *= scalar
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate(optimizer, epoch, confs):
    """Decay the learning rate based on schedule"""
    lr = confs["learning_rate"]
    if confs["cos"]:  # cosine lr schedule
        progress = float(epoch - confs["warmup-epoch"]) / float(confs["epochs"] - confs["warmup-epoch"])
        lr *= 0.5 * (1. + math.cos(math.pi * progress))
    else:  # stepwise lr schedule
        for milestone in confs["schedule"]:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_with_min(optimizer, epoch, confs):
    """Decay the learning rate based on schedule"""
    lr = confs["learning_rate"]
    if confs["cos"]:  # cosine lr schedule
        min_lr = confs["learning_rate"]
        progress = float(epoch - confs["warmup-epoch"]) / float(confs["epochs"] - confs["warmup-epoch"])
        lr = min_lr + 0.5 * (lr - min_lr) * (1. + math.cos(math.pi * progress))
    else:  # stepwise lr schedule
        for milestone in confs["schedule"]:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
