#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""
from run_config import args
from collections import OrderedDict
import os
import sys
sys.path.append(os.path.abspath("./"))

import numpy as np
import torch
from torch import nn
from timm.utils import NativeScaler

from sklearn.metrics import roc_curve, auc
import distribute as du
# from models.model import build_model
from models.focal_model import build_model
from dataset.loader import get_loaders
from utils.meters import AverageMeter,accuracy
import time
from tqdm import tqdm
import torch.distributed as dist
from dataset.dataset import Uniformer_DataSet
import utils.logger as logging

logger = logging.get_logger(__name__)
    

def my_train_epoch(
        train_loader,
        model: nn.Module,
        criterion,
        optimizer: torch.optim,
        epoch,
        sheduler: torch.optim.lr_scheduler,
        cfg
    ):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # if args.no_partialbn:
    #     model.module.partialBN(False)
    # else:
    #     model.module.partialBN(True)

    # switch to train mode
    model.train()   
    # model.module.Vedio_block.eval()

    end = time.time()
    
    # print(f"training epoch {epoch}, for total {len(train_loader)} iters: ")
    gpu_id = torch.cuda.current_device()
    message = f"[GPU {gpu_id}] training epoch {epoch}, for total {len(train_loader)} iters: "
    for i, (input, target, num_pos, embIndex, femaleAge, maleAge) in tqdm(enumerate(train_loader),desc=message):
        
        batch_size = input.size(0)

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()

        # compute output
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        embIndex = embIndex.cuda(non_blocking=True)
        femaleAge = femaleAge.cuda(non_blocking=True)
        maleAge = maleAge.cuda(non_blocking=True)

        output = model(input, embIndex, femaleAge, maleAge)
        # print("输出的size：",input_var.size())
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)
        # print(prec1)
        dist.barrier() #! 等待并同步各进程间结果
        dist.all_reduce(loss); dist.all_reduce(prec1)
        loss /= cfg.NUM_GPUS; prec1 /= cfg.NUM_GPUS
        
        
        losses.update(loss.item(), batch_size)
        top1.update(prec1.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # if args.clip_gradient is not None:
        #     total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()
        

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i+1 == len(train_loader):
    output = ('Epoch[{epoch}] GPU[{gpu_id}]:  lr: {lr:.5f}  batch_time: {batch_time.avg:.3f}  data_time: {data_time.avg:.3f}  Loss {loss.avg:.4f}  ACC-{top1.avg:.3f}'.format(
            epoch=epoch, gpu_id=gpu_id, batch_time=batch_time,
            data_time=data_time, loss=losses,top1=top1,
            lr=optimizer.param_groups[-1]['lr']
        )
    )  # TODO
    if gpu_id == 0:
        logger.info(output)
    sheduler.step()

def my_validate(val_loader, model, criterion, tf_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    # switch to evaluate mode
    device = next(model.parameters()).device
    model.eval()

    final_roc_label = []
    final_roc_pred = []

    end = time.time()
    with torch.no_grad():
        # print("validating: ")
        gpu_id = torch.cuda.current_device()
        message = f"[GPU {gpu_id}] validating :"
        for i, (input, target, num_pos, embIndex, femaleAge, maleAge) in tqdm(enumerate(val_loader),desc=message):

            final_roc_label.extend(target.tolist())
          
            batch_size = input.size(0)
            target = target.cuda()

            # compute output
            input = input.to(device)
            target = target.to(device)
            embIndex = embIndex.to(device)
            femaleAge = femaleAge.to(device)
            maleAge = maleAge.to(device)
            output = model(input, embIndex, femaleAge, maleAge)

            loss = criterion(output, target)
            losses.update(loss.item(), batch_size)

            prec1 = accuracy(output.data, target)
            top1.update(prec1.item(), batch_size)

            output_softmax = nn.Softmax(dim=-1)(output)
            probility_true = [t[1] for t in output_softmax.tolist()]
            final_roc_pred.extend(probility_true)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    fpr, tpr, th = roc_curve(np.array(final_roc_label), np.array(final_roc_pred), pos_label=1)
    roc_auc = auc(fpr, tpr)

    output = 'Validating Results: ACC-{top1.avg:.3f} ROC_AUC-{roc_auc:.3f}'.format(top1=top1, roc_auc=roc_auc)
    # log.write('----------------------------------------------\n')
    logger.info(output)

    return top1.avg, roc_auc, fpr, tpr


def my_test(cfg):
    logging.setup_logging()
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    logger.info("use GPU " + cfg.gpu_ids)
    # du.init_distributed_training(cfg)
    logger.info("use random seed " + str(cfg.RNG_SEED))
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    model = build_model(cfg)
    
    _, val_loader, _ = get_loaders(Uniformer_DataSet)
    
    log_root = os.path.join(cfg.OUTPUT_DIR,"logs")
    os.makedirs(log_root,exist_ok=True)
    # training_log = open(os.path.join(log_root,"logging.txt"),'a')
    with open(os.path.join(log_root,"args.txt"),'w') as f:
        f.write(str(cfg))
    
    start_epoch = 0

    weight_root = os.path.join(cfg.OUTPUT_DIR,"weight")
    os.makedirs(weight_root,exist_ok=True)
    prec_1, roc_auc_1, fpr, tpr = my_validate(
        val_loader=val_loader,
        model=model,
        criterion=nn.CrossEntropyLoss(),
    )
    result = OrderedDict(
        {
            "fpr": fpr,
            "tpr": tpr
        }
    )
    torch.save(result,os.path.join(weight_root,"result.pth"))
    
        
    
if __name__ == "__main__":
    my_test(args=args)
    
    