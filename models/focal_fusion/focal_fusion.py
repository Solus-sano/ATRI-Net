import torch
import torch.nn as nn
import collections
import copy
import torch.nn.functional as F
from focal_fusion.non_local_gaussian import _NonLocalBlockND
import torchvision

#这个类用来给将多维tensor 铺平为 1维tensor
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class identity(nn.Module):
    def forward(self, x):
        return x

def get_block1(model):
    return nn.Sequential(collections.OrderedDict([
        ('conv1',model.conv1),
        ('bn1',model.bn1),
        ('relu',model.relu),
        ('maxpool',model.maxpool),
        ('layer1',model.layer1)]))


# block1的参数共享
class TSM_CWA(nn.Module):
    def __init__(self, model, block1, arch="resnet34", focal_cnt = 3):
        super(TSM_CWA, self).__init__()

        self.block1 = copy.deepcopy(block1)

        numChannel = 256
        div = 16
        if arch == "resnet18" or arch == "resnet34": 
            numChannel = 64
            div = 4
        # self.att_weight = nn.Parameter(torch.randn((1,numChannel*focal_cnt)))
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(numChannel*focal_cnt, numChannel*focal_cnt//div),
            nn.ReLU(),
            nn.Linear(numChannel*focal_cnt//div, numChannel*focal_cnt)
            )
        self.down_dim = nn.Conv2d(in_channels=numChannel*focal_cnt, out_channels=numChannel, kernel_size=(1,1), bias=False)


    def forward(self,x: torch.Tensor):
        #input: (batch, focal, c, h, w)
        B, fo, c, h, w = x.shape
        batchXseg = x.size()[0]
        focal = x.size()[1]
        x = x.reshape(B*fo,c,h,w)

        y = self.block1(x)

        y = y.view((batchXseg, -1)+y.size()[-2:])#(batch, focal * c, h, w)

        channel_att_sum = None
        # channel_att_sum = self.att_weight.repeat(batchXseg,1)
        # print(y.shape)
        for pool_type in ['avg', 'max']:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( y, (y.size(2), y.size(3)), stride=(y.size(2), y.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type=='max':
                max_pool = F.max_pool2d( y, (y.size(2), y.size(3)), stride=(y.size(2), y.size(3)))
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
                
        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(y)
        
        # 残差

        y = y * scale + y
        y = self.down_dim(y)

        return y

class LSRA_fusion(nn.Module):
    def __init__(self, focal_cnt: int):
        super().__init__()
        resnet = torchvision.models.resnet34()
        self.d_model = 64
        self.res_block1 = get_block1(resnet)
        re_d_model = 64
        self.pos_embed = nn.Sequential(
            nn.Conv3d(self.d_model, re_d_model, kernel_size=1, stride=1, padding=0,groups=re_d_model),
            nn.Conv3d(self.d_model, re_d_model, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=re_d_model)
        )
        self.adapt_dim = nn.Conv2d(re_d_model*focal_cnt, self.d_model, kernel_size=1, stride=1, padding=0,bias=False)
        
    def forward(self,x: torch.Tensor):
        #input: (batch * frames, focal, c, h, w)
        B, fo, c, h, w = x.shape 
        x = x.reshape(B*fo, c, h, w)
        x = self.res_block1(x)
        x = x.reshape(B, fo, self.d_model, h//4, w//4)
        
        x = x.permute(0,2,1,3,4)
        x = self.pos_embed(x)
        B, c, fo, h, w = x.shape    

        x = x.reshape((B, c*fo, h, w))
        x = self.adapt_dim(x)
        return x
        

def get_focal_fusion(focal_cnt=3):
    net = torchvision.models.resnet34()
    return TSM_CWA(net, get_block1(net), "resnet34", focal_cnt)

class Fusion_block(nn.Module):
    def __init__(self,focal_cnt):
        super().__init__()
        # self.backbone = get_focal_fusion(focal_cnt)
        self.backbone = LSRA_fusion(focal_cnt=focal_cnt)
        
    def forward(self, x: torch.Tensor):
        """input shape: (batch, channel, focal, frames, H, W)"""
        B, C, fo, fr, H, W = x.shape
        x = x.permute(0,3,2,1,4,5) # B, fr, fo, C, H, W
        x = x.reshape(B*fr, fo, C, H, W)
        x = self.backbone(x)
        x = x.reshape(B, fr, x.shape[-3], x.shape[-2], x.shape[-1]).transpose(1,2)

        return x

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '3,4,6,7'
    net = torchvision.models.resnet34()
    # for key in net.named_modules():
    #     print(key)
    model = Fusion_block(3).to(device='cuda:0')
    x = torch.randn((4,3,3,14,224,224),device='cuda:0')
    y = model(x)
    print(y.shape)
