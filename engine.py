# Original copyright Amazon.com, Inc. or its affiliates, under CC-BY-NC-4.0 License.
# Modifications Copyright Lang Huang (laynehuang@outlook.com). All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import time
from datetime import timedelta
import faiss
import numpy as np

import torch
import torch.nn as nn

from utils import utils


def validate(val_loader, model, confs):
    losses = utils.AverageMeter('Loss', ':.4e')
    progress = utils.ProgressMeter(
        len(val_loader),
        [losses],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, caption, idx) in enumerate(val_loader):
            if confs["gpu"] is not None:
                images[0] = images[0].cuda(confs["gpu"], non_blocking=True)
                images[1] = images[0].cuda(confs["gpu"], non_blocking=True)
            
            # compute validation loss.
            with torch.cuda.amp.autocast():
                loss = model(im_v1=images[0], im_v2=images[1] , caption = caption, idx=idx)
            losses.update(loss.item() , images[0].size(0))
            if i % confs["print_freq"] == 0:
                progress.display(i)
    return loss

def ss_validate(val_loader_base, val_loader_query, model, args):
    print("start KNN evaluation with key size={} and query size={}".format(
        len(val_loader_base.dataset.samples), len(val_loader_query.dataset.samples)))
    batch_time_key = utils.AverageMeter('Time', ':6.3f')
    batch_time_query = utils.AverageMeter('Time', ':6.3f')
    # switch to evaluate mode
    model.eval()

    feats_base = []
    target_base = []
    feats_query = []
    target_query = []

    with torch.no_grad():
        start = time.time()
        end = time.time()
        # Memory features
        for i, (images, target, _) in enumerate(val_loader_base):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute features
            feats = model(images)
            # L2 normalization
            feats = nn.functional.normalize(feats, dim=1)

            feats_base.append(feats)
            target_base.append(target)

            # measure elapsed time
            batch_time_key.update(time.time() - end)
            end = time.time()

            if i % args["print_freq"] == 0:
                print('Extracting key features: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                    i, len(val_loader_base), batch_time=batch_time_key))

        end = time.time()
        for i, (images, target, _) in enumerate(val_loader_query):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute features
            feats = model(images)
            # L2 normalization
            feats = nn.functional.normalize(feats, dim=1)

            feats_query.append(feats)
            target_query.append(target)

            # measure elapsed time
            batch_time_query.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Extracting query features: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                    i, len(val_loader_query), batch_time=batch_time_query))

        feats_base = torch.cat(feats_base, dim=0)
        target_base = torch.cat(target_base, dim=0)
        feats_query = torch.cat(feats_query, dim=0)
        target_query = torch.cat(target_query, dim=0)
        feats_base = feats_base.detach().cpu().numpy()
        target_base = target_base.detach().cpu().numpy()
        feats_query = feats_query.detach().cpu().numpy()
        target_query = target_query.detach().cpu().numpy()
        feat_time = time.time() - start

        # KNN search
        index = faiss.IndexFlatL2(feats_base.shape[1])
        index.add(feats_base)
        D, I = index.search(feats_query, args.num_nn)
        preds = np.array([np.bincount(target_base[n]).argmax() for n in I])

        NN_acc = (preds == target_query).sum() / len(target_query) * 100.0
        knn_time = time.time() - start - feat_time
        print("finished KNN evaluation, feature time: {}, knn time: {}".format(
            timedelta(seconds=feat_time), timedelta(seconds=knn_time)))
        print(' * NN Acc@1 {:.3f}'.format(NN_acc))

    return NN_acc
