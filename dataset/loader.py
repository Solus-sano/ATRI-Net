from run.run_config import args
import torch
from torch.utils import data
from dataset.dataset import Uniformer_DataSet
import torchvision
from models.transforms import GroupRandom_HFlip,GroupRandom_VFlip
# from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler

cfg = args.parse_args()


def get_loaders(Dataset: data.Dataset):
    train_dataset = Dataset(
                "",
                cfg.train_lst_file,
                num_segments=cfg.num_segment,
                transform=torchvision.transforms.Compose([
                                # torchvision.transforms.Grayscale(),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.CenterCrop(224)
                        ]),
                group_transform=torchvision.transforms.Compose([
                                GroupRandom_VFlip(),
                                GroupRandom_HFlip()
                        ])
            )
    
    val_dataset = Dataset(
                "",
                cfg.val_lst_file,
                num_segments=cfg.num_segment,
                transform=torchvision.transforms.Compose([
                                # torchvision.transforms.Grayscale(),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.CenterCrop(224)
                        ]),
            )
    
    test_dataset = Dataset(
                "",
                cfg.test_lst_file,
                num_segments=cfg.num_segment,
                transform=torchvision.transforms.Compose([
                                # torchvision.transforms.Grayscale(),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.CenterCrop(224)
                        ]),
            )
    
    if cfg.NUM_GPUS>1: 
        train_loader = data.DataLoader(
            train_dataset,
            cfg.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=cfg.pin_memory,
            sampler=DistributedSampler(train_dataset)
        )
    else:
        train_loader = data.DataLoader(
            train_dataset,
            cfg.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=cfg.pin_memory
        )
    val_loader = data.DataLoader(
        val_dataset,
        cfg.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=cfg.pin_memory
    )
    
    test_loader = data.DataLoader(
        test_dataset,
        cfg.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=cfg.pin_memory
    )

    return train_loader, val_loader, test_loader