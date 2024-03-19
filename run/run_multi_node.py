#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""
# from slowfast.config.defaults import assert_and_infer_cfg
from distribute import launch_job
# from slowfast.utils.parser import load_config, parse_args
from run_config import args

# from demo_net import demo
# from test_net import test
from train_net import my_train
from test_net import my_test
# from train_net_feature import my_train
# from visualization import visualize

import os


# def parse_ip(s):
#     s = s.split("-")
#     s = [y for x in s for y in x.split("[") if y]
#     s = [y for x in s for y in x.split(",") if y ]

#     return ".".join(s[2:6])


def main():
    """
    Main function to spawn the train and test process.
    """
    # cfg = args.parse_args()

    # if 'SLURM_STEP_NODELIST' in os.environ:
    #     args.init_method = "tcp://{}:{}".format(
    #         parse_ip(os.environ['SLURM_STEP_NODELIST']), "9999")

    #     print("Init Method: {}".format(args.init_method))

    #     cfg = load_config(args)
    #     cfg = assert_and_infer_cfg(cfg)

    #     cfg.NUM_SHARDS = int(os.environ['SLURM_NTASKS'])
    #     cfg.SHARD_ID = int(os.environ['SLURM_NODEID'])

    #     print(f'node id > {cfg.SHARD_ID}')
    # else:
    #     cfg = load_config(args)
    #     cfg = assert_and_infer_cfg(cfg)

    # Perform training.
    # if cfg.TRAIN.ENABLE:
    cfg = args.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    launch_job(cfg=cfg, func=my_train)

    # Perform multi-clip testing.
    # if cfg.TEST.ENABLE:
    #     launch_job(cfg=cfg, init_method=args.init_method, func=test)

    # Perform model visualization.
    # if cfg.TENSORBOARD.ENABLE and (
    #     cfg.TENSORBOARD.MODEL_VIS.ENABLE
    #     or cfg.TENSORBOARD.WRONG_PRED_VIS.ENABLE
    # ):
    #     launch_job(cfg=cfg, init_method=args.init_method, func=visualize)

    # Run demo.
    # if cfg.DEMO.ENABLE:
    #     demo(cfg)


if __name__ == "__main__":
    main()
