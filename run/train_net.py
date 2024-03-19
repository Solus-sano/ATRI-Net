#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""
from run_config import args
import os
import sys
sys.path.append(os.path.abspath("./"))

from collections import OrderedDict
import numpy as np
import pprint
import torch
from torch import nn
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
from timm.utils import NativeScaler

from sklearn.metrics import roc_curve, auc
from ckpt_utils import save_checkpoint,save_best_checkpoint
import distribute as du
# from models.model import build_model
from models.focal_model import build_model
from dataset.loader import get_loaders
from utils.meters import AverageMeter,accuracy
import time
from tqdm import tqdm
import torch.distributed as dist
from dataset.dataset import Uniformer_DataSet,Uniformer_Feature_DataSet
# import slowfast.models.losses as losses
# import slowfast.models.optimizer as optim
# import slowfast.utils.checkpoint_amp as cu
# import slowfast.utils.distributed as du
# import slowfast.utils.logging as logging
import utils.logger as logging
# import slowfast.utils.metrics as metrics
# import slowfast.utils.misc as misc
# import slowfast.visualization.tensorboard_vis as tb
# from slowfast.datasets import loader
# from slowfast.datasets.mixup import MixUp
# from slowfast.models import build_model
# from slowfast.utils.meters import AVAMeter, EpochTimer, TrainMeter, ValMeter
# from slowfast.utils.multigrid import MultigridSchedule

logger = logging.get_logger(__name__)



# def print_GPU_info():
#     gpu_memory_info = torch.cuda.memory_allocated()
#     print("Current GPU[{}] Memory Usage:".format(torch.cuda.current_device()))
#     print(f"Allocated: {gpu_memory_info/1024**2:.2f} MB")
#     print(f"Cached: {gpu_memory_info/1024**2:.2f} MB")
    

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

def my_validate(val_loader, model, criterion, epoch, tf_writer=None):
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
        message = f"[GPU {gpu_id}] validating epoch {epoch}:"
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

    output = f"epoch [{epoch}]: "
    output += (
        'Validating Results: ACC-{top1.avg:.3f} ROC_AUC-{roc_auc:.3f}'.format(top1=top1, roc_auc=roc_auc))
    # log.write('----------------------------------------------\n')
    logger.info(output)

    return top1.avg, roc_auc, fpr, tpr


def my_train(cfg):
    logging.setup_logging()
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    logger.info("use GPU " + cfg.gpu_ids)
    # du.init_distributed_training(cfg)
    logger.info("use random seed " + str(cfg.RNG_SEED))
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    model = build_model(cfg)
    policy = model.module.get_optim_policies()
    if torch.cuda.current_device()==0:
        for group in policy:
            p_cnt = sum(i.numel() for i in group["params"])
            logger.info("param group {} has {} params, lr_mult: {}".format(group["name"],p_cnt,group["lr_mult"]))
    # updater = torch.optim.Adam(
    #     policy,
    #     cfg.lr,
    #     weight_decay=cfg.weight_decay
    # )
    updater = torch.optim.AdamW(
        policy,
        cfg.lr,
        [0.9, 0.999],
        weight_decay=cfg.weight_decay
    )
    sheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=updater,
        T_max=cfg.epochs,
        eta_min=cfg.lr*0.1
    )
    logger.info("use CosineAnnealingLR, lr_begin={}, lr_end={}".format(cfg.lr,cfg.lr*0.1))
    # sheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer=updater,
    #     step_size=50,
    #     gamma=0.1,
    # )
    # logger.info("use stepLR, step_size={}, gamma={}".format(40,0.1))
    
    train_loader, val_loader, test_loader = get_loaders(Uniformer_DataSet)
    
    
    log_root = os.path.join(cfg.OUTPUT_DIR,"logs")
    os.makedirs(log_root,exist_ok=True)
    # training_log = open(os.path.join(log_root,"logging.txt"),'a')
    with open(os.path.join(log_root,"args.txt"),'w') as f:
        f.write(str(cfg))
        
    best_prec_epoch = 0
    best_roc_auc = 0
    best_roc_epoch = 0
    best_prec = 0
    
    start_epoch = 0
    # model.module.freeze_Vedio_block()
    for epoch in range(start_epoch,cfg.epochs):
        # if epoch == 45:
        #     model.module.unfreeze_Vedio_block()
        my_train_epoch(
            train_loader=train_loader,
            model=model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=updater,
            epoch=epoch,
            sheduler=sheduler,
            cfg=cfg,
        )
        
        is_checkp_epoch = (epoch)%cfg.ckpt_freq==0
        is_eval_epoch = epoch%cfg.eval_freq==0
        
        if is_eval_epoch and torch.cuda.current_device()==0:
            weight_root = os.path.join(cfg.OUTPUT_DIR,"weight")
            os.makedirs(weight_root,exist_ok=True)
            prec_1, roc_auc_1, fpr, tpr = my_validate(
                val_loader=val_loader,
                model=model,
                criterion=nn.CrossEntropyLoss(),
                epoch=epoch,
            )
            
            # prec_2, roc_auc_2 = my_validate(
            #     val_loader=test_loader,
            #     model=model,
            #     criterion=nn.CrossEntropyLoss(),
            #     epoch=epoch,
            # )
            
            if prec_1 > best_prec:
                torch.save(model.module.state_dict(),os.path.join(weight_root,f"best_prec.pth"))
                best_prec_epoch = epoch
                best_prec = prec_1
            if roc_auc_1 > best_roc_auc:
                torch.save(model.module.state_dict(),os.path.join(weight_root,f"best_auc.pth"))
                roc_result = OrderedDict(
                    {"fpr": fpr, "tpr": tpr}
                )
                torch.save(roc_result,os.path.join(weight_root,f"roc_result.pth"))
                best_roc_epoch = epoch
                best_roc_auc = roc_auc_1
                
            logger.info(f"best ACC: {best_prec:.3f} (epoch: {best_prec_epoch})")
            logger.info(f"best AUC: {best_roc_auc:.3f} (epoch: {best_roc_epoch})\n")
            
            if is_checkp_epoch:
                torch.save(model.module.state_dict(),os.path.join(weight_root,f"epoch{epoch}.pth"))
        
    
if __name__ == "__main__":
    my_train(args=args)
    
    