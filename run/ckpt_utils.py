import os
import distribute as du
import torch
import copy


def save_checkpoint(path_to_job, model, optimizer, loss_scaler, epoch, cfg, zip=True):
    """
    Save a checkpoint.
    Args:
        model (model): model to save the weight to the checkpoint.
        optimizer (optim): optimizer to save the historical state.
        loss_scaler (scaler): scaler for loss.
        epoch (int): current number of epoch of the model.
        cfg (CfgNode): configs to save.
    """
    # Save checkpoints only from the master process.
    if not du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        return
    
    ckpt_dir = os.path.join(cfg.OUTPUT_DIR,"ckpt")
    # Ensure that the checkpoint dir exists.
    os.makedirs(ckpt_dir,exist_ok=True)
    # Omit the DDP wrapper in the multi-gpu setting.
    sd = model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict()
    normalized_sd = sub_to_normal_bn(sd)

    # Record the state.
    checkpoint = {
        "epoch": epoch,
        "model_state": normalized_sd,
        "optimizer_state": optimizer.state_dict(),
        'scaler': loss_scaler.state_dict(),
        "cfg": cfg.dump(),
    }
    # Write the checkpoint.
    # path_to_checkpoint = get_path_to_checkpoint(path_to_job, epoch + 1, save_latest=cfg.TRAIN.SAVE_LATEST)
    path_to_checkpoint = os.path.join(ckpt_dir,"epoch{}.ckpt")
    # with g_pathmgr.open(path_to_checkpoint, "wb") as f:
    #     if not zip:
    #         torch.save(checkpoint, f, _use_new_zipfile_serialization=False)
    #     else:
    #         torch.save(checkpoint, f)
    torch.save(checkpoint,path_to_checkpoint)
    return path_to_checkpoint

def save_best_checkpoint(path_to_job, model, optimizer, loss_scaler, epoch, cfg, zip=True):
    """
    Save a checkpoint.
    Args:
        model (model): model to save the weight to the checkpoint.
        optimizer (optim): optimizer to save the historical state.
        loss_scaler (scaler): scaler for loss.
        epoch (int): current number of epoch of the model.
        cfg (CfgNode): configs to save.
    """
    # Save checkpoints only from the master process.
    if not du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        return
    
    ckpt_dir = os.path.join(cfg.OUTPUT_DIR,"ckpt")
    # Ensure that the checkpoint dir exists.
    os.makedirs(ckpt_dir,exist_ok=True)
    # Omit the DDP wrapper in the multi-gpu setting.
    sd = model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict()
    normalized_sd = sub_to_normal_bn(sd)

    # Record the state.
    checkpoint = {
        "epoch": epoch,
        "model_state": normalized_sd,
        "optimizer_state": optimizer.state_dict(),
        'scaler': loss_scaler.state_dict(),
        "cfg": cfg.dump(),
    }
    # Write the checkpoint.
    # path_to_checkpoint = get_path_to_checkpoint(path_to_job, epoch + 1, save_latest=cfg.TRAIN.SAVE_LATEST)
    path_to_checkpoint = os.path.join(ckpt_dir,"best.ckpt")
    # with g_pathmgr.open(path_to_checkpoint, "wb") as f:
    #     if not zip:
    #         torch.save(checkpoint, f, _use_new_zipfile_serialization=False)
    #     else:
    #         torch.save(checkpoint, f)
    torch.save(checkpoint,path_to_checkpoint)
    return path_to_checkpoint

def sub_to_normal_bn(sd):
    """
    Convert the Sub-BN paprameters to normal BN parameters in a state dict.
    There are two copies of BN layers in a Sub-BN implementation: `bn.bn` and
    `bn.split_bn`. `bn.split_bn` is used during training and
    "compute_precise_bn". Before saving or evaluation, its stats are copied to
    `bn.bn`. We rename `bn.bn` to `bn` and store it to be consistent with normal
    BN layers.
    Args:
        sd (OrderedDict): a dict of parameters whitch might contain Sub-BN
        parameters.
    Returns:
        new_sd (OrderedDict): a dict with Sub-BN parameters reshaped to
        normal parameters.
    """
    new_sd = copy.deepcopy(sd)
    modifications = [
        ("bn.bn.running_mean", "bn.running_mean"),
        ("bn.bn.running_var", "bn.running_var"),
        ("bn.split_bn.num_batches_tracked", "bn.num_batches_tracked"),
    ]
    to_remove = ["bn.bn.", ".split_bn."]
    for key in sd:
        for before, after in modifications:
            if key.endswith(before):
                new_key = key.split(before)[0] + after
                new_sd[new_key] = new_sd.pop(key)

        for rm in to_remove:
            if rm in key and key in new_sd:
                del new_sd[key]

    for key in new_sd:
        if key.endswith("bn.weight") or key.endswith("bn.bias"):
            if len(new_sd[key].size()) == 4:
                assert all(d == 1 for d in new_sd[key].size()[1:])
                new_sd[key] = new_sd[key][:, 0, 0, 0]

    return new_sd