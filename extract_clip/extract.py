import clip
import torch
from collections import OrderedDict
import os
root = "/data2/liangzhijia/ckpt"

model, _ = clip.load("ViT-B/16", device='cpu')
new_state_dict = OrderedDict()
for k, v in model.state_dict().items():
    if 'visual.' in k:
        if k[7:] not in ["proj", "ln_post.weight", "ln_post.bias"]:
            new_state_dict[k[7:]] = v
torch.save(new_state_dict, os.path.join(root,'vit_b16.pth'))

model, _ = clip.load("ViT-L/14", device='cpu')
new_state_dict = OrderedDict()
for k, v in model.state_dict().items():
    if 'visual.' in k:
        if k[7:] not in ["proj", "ln_post.weight", "ln_post.bias"]:
            new_state_dict[k[7:]] = v
torch.save(new_state_dict, os.path.join(root,'vit_l14.pth'))

model, _ = clip.load("ViT-L/14@336px", device='cpu')
new_state_dict = OrderedDict()
for k, v in model.state_dict().items():
    if 'visual.' in k:
        if k[7:] not in ["proj", "ln_post.weight", "ln_post.bias"]:
            new_state_dict[k[7:]] = v
torch.save(new_state_dict, os.path.join(root,'vit_l14_336.pth'))