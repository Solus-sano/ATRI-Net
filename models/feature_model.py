import os
import sys
sys.path.append(os.path.abspath("./models"))
from collections import OrderedDict
import yaml

import random
from torch import nn
import torch
import torchvision
from torchvision import transforms
# from ops.basic_ops import ConsensusModule
# from transfroms import *
# from ops.focal_fusion import *
from mutan import EggsDataNet,MutanFusion
from torch.nn.init import normal_, constant_
from torch.nn.parallel import DistributedDataParallel as DDP
import utils.logger as logging
import torch.distributed as dist
logger = logging.get_logger(__name__)

def param_cnt(net: nn.Module):
    cnt = sum(p.numel() for p in net.parameters())
    op = None
    if cnt >1e+9:
        op = 'model num parameters: ' + str(cnt//1e+9) + 'B'
    elif cnt>1e+6:
        op = 'model num parameters: ' + str(cnt//1e+6) + 'M'
    elif cnt>1e+3:
        op = 'model num parameters: ' + str(cnt//1e+3) + 'K'
    else:
        op = 'model num parameters: ' + str(cnt)
    return op
        
def build_model(cfg):
    """prepare and return a model"""
    net = Uniformer_ART(
        num_class=cfg.num_class,
        num_segments=cfg.num_segment,
        modality=cfg.modality,
        cfg=cfg
    )
    if torch.cuda.current_device() == 0:
        for item in net.children():
            logger.info(str(item))
    
    current_device = "cuda:" + str(torch.cuda.current_device()) if cfg.NUM_GPUS>0 else "cpu"
    # print("load net to device " + current_device)
    # if torch.cuda.current_device()==0: os.system("nvidia-smi")
    net = net.to(current_device)
    
    if cfg.NUM_GPUS>1:
        net = DDP(
            module=net,
            device_ids=[torch.cuda.current_device()],
            output_device=torch.cuda.current_device(),
            find_unused_parameters=True
        )
    if torch.cuda.current_device()==0:
        if cfg.NUM_GPUS>1:
            logger.info(param_cnt(net.module))
        else:
            logger.info(param_cnt(net))
    return net

class PositionalEncoding(nn.Module):
    def __init__(self,
            num_hiddens: int,
            dropout: float,
            max_token_len = 77    
        ) :
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_token_len, num_hiddens))

        tmp_col = torch.arange(max_token_len,dtype=torch.float32).reshape(-1,1)
        tmp_row = torch.pow(10000, torch.arange(0,num_hiddens,2, dtype=torch.float32) / num_hiddens)
        tmp_map = tmp_col / tmp_row

        self.P[:,:,::2] = torch.sin(tmp_map)
        self.P[:,:,1::2] = torch.cos(tmp_map)

    def forward(self, X: torch.Tensor):
        return self.dropout(X + self.P[:,:X.shape[1],:].to(X.device))

class Attention(nn.Module):
    def __init__(self,num_dim=1024,num_head=4):
        super().__init__()
        self.num_dim = num_dim
        self.num_head = num_head
        self.tmp_token = nn.Parameter(torch.randn((1,1,num_dim)))
        self.qkv_proj = nn.Linear(num_dim,num_dim*3)
        self.v_proj = nn.Linear(num_dim,num_dim)
        self.att_score_dropout = nn.Dropout(0.5)
        self.op_dropout = nn.Dropout(0.5)
        for p in self.v_proj.parameters():
            nn.init.zeros_(p)
        
    def forward(self,input: torch.Tensor):
        x = torch.cat([self.tmp_token.repeat(input.shape[0],1,1), input],dim=1)
        B, N, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_head, C // self.num_head)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, bs, heads, N, dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        att_score = (q @ k.transpose(-2,-1)) * ( (C // self.num_head) ** -0.5 )
        att_score = att_score.softmax(dim=-1)
        att_score = self.att_score_dropout(att_score)
        
        op = (att_score @ v).permute(0, 2, 1, 3).reshape(B, N, C)
        op = self.v_proj(op)
        
        return self.op_dropout(op)[:,0,:]
        

class Uniformer_ART(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 new_length=None, cfg=None,
                 consensus_type='identity', before_softmax=True,
                 dropout=0.8,
                 partial_bn=True, pretrain='imagenet',
                 is_shift=False, shift_place='blockres', fc_lr5=False,
                 temporal_pool=False, non_local=False):
        super().__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.consensus_type = consensus_type
        self.pretrain = pretrain

        self.is_shift = is_shift
        self.shift_place = shift_place
        self.fc_lr5 = fc_lr5
        self.temporal_pool = temporal_pool
        self.non_local = non_local
        
        self.pretrain_param_name_lst = []
        self.finetune_param_name_lst = []

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        # 临床数据embedding部分 
        #! 输入维度
        self.emb_inDim = 32
        self.emb_outDim = 32
        self.feature_dropout_1 = nn.Dropout(0.5)
        self.feature_dropout_2 = nn.Dropout(0.5)
        # self.FFN = nn.Linear(1024*2,1024)
        # self.pos_emb = PositionalEncoding(1024,0.5)
        # self.Vedion_emb_att = Attention()
        # for p in self.Vedion_emb_att.v_proj.parameters():
        #     nn.init.zeros_(p)
        
        self.embedding = EggsDataNet(self.emb_inDim, self.emb_outDim)

        # mutan多模态融合部分 
        self.mutan_data_inDim = 32
        self.mutan_vedio_inDim = 1024
        self.mutan_outDim = 32
        self.mutan_layers=5
        #! 修改了输入维度，因为data和Uniformer抽取的vedio编码长度不一致的情况
        self.mutan = MutanFusion(self.mutan_vedio_inDim, self.mutan_data_inDim, self.mutan_outDim, self.mutan_layers)
        # mutan的输入是 bi-LSTM的输出向量 以及临床数据的embedding向量 输出与这些向量维度一样的向量 

        self.final_fc = nn.Linear(self.mutan_outDim, num_class)
        self.dropout_layer = nn.Dropout(p=0.5)

        '''添加部分到此结束'''
    
    def train(self, mode=True):
        super().train(mode)
        # count = 0
        # if self._enable_pbn and mode:
        #     print("Freezing BatchNorm2D except the first one.")
        #     for m in self.base_model.modules():
        #         if isinstance(m, nn.BatchNorm2d):
        #             count += 1
        #             #只有7个block1的bn会更新
        #             if count >= (2 if self._enable_pbn else 1):
        #                 m.eval()
        #                 m.weight.requires_grad = False
        #                 m.bias.requires_grad = False
    
    def get_optim_policies(self):
        # self.set_finetune_params()
        emb_param_lst = []
        mutan_param_lst = []
        other_param_lst = []
        
        for name, m in self.named_modules():
            if "embedding" in name:
                for name,p in m.named_parameters(recurse=False):
                    emb_param_lst.append(p)
            elif "mutan" in name:
                for name,p in m.named_parameters(recurse=False):
                    mutan_param_lst.append(p)
            else:
                for name,p in m.named_parameters(recurse=False):
                    other_param_lst.append(p)
            
        
        return [
            {"params": emb_param_lst, "lr_mult": 1, "name": "emb_param"},
            {"params": mutan_param_lst, "lr_mult": 1, "name": "mutan_param"},
            {"params": other_param_lst, "lr_mult": 1, "name": "other_param"}
        ]

    def forward(self, feature, emb_index, femaleAge, maleAge):
        # main_video_output = feature[:,3,:]
        feature = feature[:,3:4,:]
        B, fo, C = feature.shape
        # feature = self.feature_dropout_1(feature)
        # video_output = self.Vedion_emb_att(self.pos_emb(feature))
        video_output = feature.reshape(B,-1)
        # if self.training:
        #     if random.random() < 0.1:
        #         video_output = main_video_output 
        #     else:
        #         video_output = main_video_output + video_output
        # else:
        #     video_output = main_video_output + video_output
        video_output = self.feature_dropout_2(video_output)
        
        data_output = self.embedding(emb_index, femaleAge, maleAge)
        
        mutan_output = self.mutan(video_output, data_output)
        final_output = self.dropout_layer(mutan_output)
        final_output = self.final_fc(final_output)
    
        return final_output
        
    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224
 
            
if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "2,3,6,7"
    device = 'cuda'
    
    # net = Uniformer_ART(2,16,'RGB').to(device)
    # print(param_cnt(net))
    # X = torch.rand((1,3,3,16,224,224)).to(device)
    # Y = net(X,torch.Tensor([[0,1,2,5]]).to(torch.int).to(device),torch.Tensor([0.4]).to(device),torch.Tensor([0.3]).to(device))
    # print(Y.shape)
    
    # with open("Unit_ART.txt",'w') as f:
    #     for name,m in net.named_modules():
    #         f.write(name + "\n")
    
    att_net = Attention()
    x = torch.rand((2,3,1024))
    print(att_net(x).shape)
        
    
