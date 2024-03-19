#!/usr/bin/env python3
import argparse
args = argparse.ArgumentParser()

args.add_argument("--RNG_SEED",type=int,default=114,help="global random seed")
args.add_argument("--OUTPUT_DIR",type=str,default="/data2/liangzhijia/Blastocyst/Uniformerv2_ART/run/train_record/train_61")
args.add_argument("--ckpt_freq",type=int,default=20000,help="frequence to get checkpoint")
args.add_argument("--eval_freq",type=int,default=1,help="frequence to eval")

args.add_argument("--gpu_ids",type=str,default="0,1,3,4")
args.add_argument("--NUM_GPUS",type=int,default=4)
# args.add_argument("--SHARD_ID",type=int,default=0)

args.add_argument("--epochs",type=int,default=250)
args.add_argument("--batch_size",type=int,default=8,help="the batch_size is used for each GPU(not the total batch_size)")
args.add_argument("--train_lst_file",type=str,default="/data2/liangzhijia/Blastocyst/Uniformerv2_ART/data_opt/train_data_clean.txt")
args.add_argument("--val_lst_file",type=str,default="/data2/liangzhijia/Blastocyst/Uniformerv2_ART/data_opt/test_data_clean.txt")
args.add_argument("--test_lst_file",type=str,default="/data2/liangzhijia/Blastocyst/Uniformerv2_ART/data_opt/test_data_clean.txt")

"""model config"""
args.add_argument("--num_class",type=int,default=2)
args.add_argument("--num_segment",type=int,default=16)
args.add_argument("--Vedio_backbone",type=str,default="sth2_b16_16x224")
args.add_argument("--modality",type=str,default="RGB")
args.add_argument("--use_ckpt",type=bool,default=False)
args.add_argument("--ckpt_file",type=str,default="/data2/liangzhijia/Blastocyst/Uniformerv2_ART/run/train_record/train_38/weight/best_auc.pth")

"""learning config"""
args.add_argument("--lr",type=float,default=1e-3,help="learning rate")
args.add_argument("--momentum",type=float,default=0.9)
args.add_argument("--weight_decay",type=float,default=0.05)

"""dataset config"""
args.add_argument("--pin_memory",type=bool,default=True)
