
import os
import time

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed 

import backbone as backbone_models
from models import get_lewel_model
from utils import utils, lr_schedule, LARS, get_norm
import data.transforms as data_transforms
from engine import ss_validate,validate
from data.base_dataset import get_dataset
backbone_model_names = sorted(name for name in backbone_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(backbone_models.__dict__[name]))

confs = {
    "local_rank":0,
    "dataset" : "newsclippings",
    "data-root":r"E:/Chinmay/news_clippings",
    #"data-root":r'/projects/academic/sreyasee/msmu',
    "arch":"LEWELB",
    "backbone":"clip_encoder",
    "workers":8,
    "epochs": 150,
    "start-epoch": 0,
    "warmup-epoch":10,
    "batch-size":2,
    "learning_rate":1e-3,
    "schedule":[120,160],
    "cos":True,
    "momentum":0.9,
    "weight_decay": 1e-5,
    "save-dir":r'/projects/academic/sreyasee/chinmayd/saved_models',
    "print_freq":200,
    "save_freq":10,
    "eval_freq":1,
    "resume":None,
    "pretrained":"",
    "super-pretrained":"",
    "evaluate":False,
    "world-size":1,
    "rank":0,
    "dist-url":"tcp://224.66.41.62:23456",
    "dist-backend":"nccl",
    "seed":"23456",
    "gpu":None,
    "port":5389,
    "multiprocessing-distributed": None,
    "proj_dim":256,
    "enc-m":0.996,
    "norm":None, 
    "num-neck-mlp":2,
    "hid-dim":4096,
    "amp": None, 
    "lewel-l2-norm": True,
    "lewel-scale":1,
    "lewel-num-heads":4,
    "lewel-loss-weight":0.5,
    "num-nn": 20 ,
    "nn-mem-percent":0.1,
    "nn-query-percent":0.5,
    "scale":1
}

torch.cuda.device_count()

def main():
    best_val_loss = float('inf')
    # create model
    print("=> creating model '{}' with backbone '{}'".format(confs["arch"], confs["backbone"]))
    model_func = get_lewel_model(confs["arch"])
    norm_layer = get_norm(confs["norm"])
    model = model_func(
        dim=confs["proj_dim"],
        m=confs["enc-m"],
        hid_dim=confs["hid-dim"],
        num_neck_mlp=confs["num-neck-mlp"],
        scale=confs["scale"],
        l2_norm=confs["lewel-l2-norm"],
        num_heads=confs["lewel-num-heads"],
        loss_weight=confs["lewel-loss-weight"],
    )
    print(model)
    print(confs)

    if confs["pretrained"]:
        if os.path.isfile(confs["pretrained"]):
            print("=> loading pretrained model from '{}'".format(confs["pretrained"]))
            state_dict = torch.load(confs["pretrained"], map_location="cpu")['state_dict']
            # rename state_dict keys
            for k in list(state_dict.keys()):
                new_key = k.replace("module.", "")
                state_dict[new_key] = state_dict[k]
                del state_dict[k]
            msg = model.load_state_dict(state_dict, strict=False)
            print("=> loaded pretrained model from '{}'".format(confs["pretrained"]))
            if len(msg.missing_keys) > 0:
                print("missing keys: {}".format(msg.missing_keys))
            if len(msg.unexpected_keys) > 0:
                print("unexpected keys: {}".format(msg.unexpected_keys))
        else:
            print("=> no pretrained model found at '{}'".format(confs["pretrained"]))


    if torch.cuda.is_available():
        model.cuda()
    
    confs["batch-size"] = int(confs["batch-size"]/ confs["world-size"])
    confs["workers"] = int((confs["workers"] + confs["world-size"] - 1) / confs["world-size"])

    # define optimizer
    params = collect_params(model, exclude_bias_and_bn=True, sync_bn='EMAN' in confs["arch"])
    optimizer = LARS(params, lr=confs["learning_rate"], momentum=confs["momentum"], weight_decay=confs["weight_decay"])
    scaler = torch.cuda.amp.GradScaler() if confs["amp"] else None


    # optionally resume from a checkpoint
    if confs["resume"]:
        if os.path.isfile(confs["resume"]):
            print("=> loading checkpoint '{}'".format(confs["resume"]))
            if confs["gpu"] is None:
                checkpoint = torch.load(confs["resume"])
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda'
                checkpoint = torch.load(confs["resume"], map_location=loc)
            confs["start-epoch"] = checkpoint['epoch']
            if 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(confs["resume"], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(confs["resume"]))

    cudnn.benchmark = True
    # Data loading code
    transform1, transform2 = data_transforms.get_byol_tranforms()
    train_dataset = get_dataset(
        confs["dataset"],
        mode='train',
        transform=data_transforms.TwoCropsTransform(transform1, transform2),
        data_root=confs["data-root"])
    print("train_dataset:\n{}".format(len(train_dataset)))
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=confs["batch-size"], shuffle=(train_sampler is None),
        num_workers=confs["workers"], pin_memory=True, sampler=train_sampler, drop_last=True,
        persistent_workers=True)
    
    val_dataset = get_dataset(
            confs["dataset"],
            mode='eval',
            transform=data_transforms.TwoCropsTransform(transform1, transform2),
            data_root=confs["data-root"],
        ),
    print('Validation dataset' , len(val_dataset))
    val_loader_base = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=confs["batch-size"], shuffle=False,
        num_workers=confs["workers"]//2, pin_memory=True,
        persistent_workers=True)
    
    val_loader_query = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=confs["batch-size"], shuffle=False,
        num_workers=confs["workers"]//2, pin_memory=True,
        persistent_workers=True)

    if confs["evaluate"]:
        ss_validate(val_loader_base, val_loader_query, model, confs)
        return

    best_epoch = confs["start-epoch"]
    print('Start tfhe training')
    epoch = 0
    
    for epoch in range(confs["start-epoch"], confs["epochs"]):
        if epoch >= confs["warmup-epoch"]:
            lr_schedule.adjust_learning_rate(optimizer, epoch, confs)

        # train for one epoch
        train(train_loader, model, optimizer, scaler, epoch, confs)
        if (epoch + 1) % confs["eval_freq"] == 0:
            val_loss = validate(val_loader_base , model,confs)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
        
        if not confs["multiprocessing-distributed"] or (confs["multiprocessing-distributed"] and 
        confs["local_rank"] % confs["world-size"] == 0):
            utils.save_checkpoint({
            'epoch': epoch + 1,
            'arch': confs["arch"],
            'state_dict': model.state_dict(),
            'best_val_loss': best_val_loss,
            'optimizer': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
        }, epoch=epoch, confs=confs)

    print('Best Val loss {0} @ epoch {1}'.format(best_val_loss, best_epoch + 1))

def train(train_loader, model, optimizer, scaler, epoch, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    curr_lr = utils.InstantMeter('LR', '')
    curr_mom = utils.InstantMeter('MOM', '')
    progress = utils.ProgressMeter(
        len(train_loader),
        [curr_lr, curr_mom, batch_time, data_time, losses],
        prefix="Epoch: [{}/{}]\t".format(epoch, confs["epochs"]))

    # iter info
    batch_iter = len(train_loader)
    max_iter = float(batch_iter * confs["epochs"])
    # switch to train mode
    model.train()
    if "EMAN" in confs["arch"]:
        print("setting the key model to eval mode when using EMAN")
        if hasattr(model, 'module'):
            model.module.target_net.eval()
        else:
            model.target_net.eval()

    end = time.time()
    for i, (images, caption, idx) in enumerate(train_loader):
        # update model momentum
        curr_iter = float(epoch * batch_iter + i)

        # measure data loading time
        data_time.update(time.time() - end)

        if confs["gpu"] is not None:
            images[0] = images[0].cuda(confs["gpu"], non_blocking=True)
            images[1] = images[1].cuda(confs["gpu"], non_blocking=True)
            idx = idx.cuda(confs["gpu"], non_blocking=True)

        # warmup learning rate
        if epoch < confs["warmup-epoch"]:
            warmup_step = confs["warmup-epoch"] * batch_iter
            curr_step = epoch * batch_iter + i + 1
            lr_schedule.warmup_learning_rate(optimizer, curr_step, warmup_step, confs)
        curr_lr.update(optimizer.param_groups[0]['lr'])

        if scaler is None:
            # compute loss
            with torch.cuda.amp.autocast():
                loss = model(im_v1=images[0], im_v2=images[1] , caption = caption, idx=idx)
            # measure accuracy and record loss
            losses.update(loss.item(), images[0].size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:   # AMP
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model(im_v1=images[0], im_v2=images[1], idx=idx)
            # measure accuracy and record loss
            losses.update(loss.item(), images[0].size(0))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        if hasattr(model, 'module'):
            model.module.momentum_update(curr_iter, max_iter)
            curr_mom.update(model.module.curr_m)
        else:
            model.momentum_update(curr_iter, max_iter)
            curr_mom.update(model.curr_m)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % confs["print_freq"] == 0:
            progress.display(i)

def collect_params(model, exclude_bias_and_bn=True, sync_bn=True):
    """
    exclude_bias_and bn: exclude bias and bn from both weight decay and LARS adaptation
        in the PyTorch implementation of ResNet, `downsample.1` are bn layers
    """
    weight_param_list, bn_and_bias_param_list = [], []
    weight_param_names, bn_and_bias_param_names = [], []
    for name, param in model.named_parameters():
        if exclude_bias_and_bn and ('bn' in name or 'downsample.1' in name or 'bias' in name or (sync_bn and 'mlp.1' in name)):
            bn_and_bias_param_list.append(param)
            bn_and_bias_param_names.append(name)
        else:
            weight_param_list.append(param)
            weight_param_names.append(name)
    print("weight params:\n{}".format('\n'.join(weight_param_names)))
    print("bn and bias params:\n{}".format('\n'.join(bn_and_bias_param_names)))
    param_list = [{'params': bn_and_bias_param_list, 'weight_decay': 0., 'lars_exclude': True},
                  {'params': weight_param_list}]
    return param_list

if __name__ == '__main__':
    
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    confs["distributed"] = False
    confs["multiprocessing_distributed"] = False
    cudnn.benchmark = True

    main()


