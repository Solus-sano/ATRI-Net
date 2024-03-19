import os
import sys
sys.path.append(os.path.abspath("./models"))
from collections import OrderedDict
import yaml

from focal_fusion.focal_fusion import Fusion_block
from torch import nn
import torch
from focal_fusion.focal_fusion import get_focal_fusion
# from ops.basic_ops import ConsensusModule
from Uniformerv2 import uniformerv2_b16,uniformerv2_l14
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
            find_unused_parameters=False
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


        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

        '''以下是加上bi-LSTM、临床数据embedding、以及MUTAN的初始化数据'''
        # self.relu = nn.ReLU()
        # # bi-LSTM部分 
        # self.LSTM_inDim = 32
        # self.LSTM_hidDim = 16
        # self.LSTM_layers = 1 #LSTM的层数为1层 
        # self.LSTM_dirs = 2 #双向LSTM
        # self.LSTM = nn.LSTM(input_size=self.LSTM_inDim, hidden_size=self.LSTM_hidDim, num_layers=self.LSTM_layers, bias=True, 
        #                     batch_first=False, bidirectional=(True if self.LSTM_dirs == 2 else False))
        # # 输入LSTM的特征维度是 (batch_size, num_segments, LSTM_inDim) 要先将0和1维度对换 
        # # LSTM输出的维度是 (num_segments, batch, LSTM_hidDim*2) 
        
        #! 使用Uniformerv2抽取视频特征
        
        with open("/data2/liangzhijia/Blastocyst/Uniformerv2_ART/models/Uniformer_cfg.yaml",'r') as f:
            yaml_dict = yaml.safe_load(f)
            self.Vedio_block_ckpt_file = yaml_dict[cfg.Vedio_backbone]["ckpt"]
            self.Vedio_block_cfg = yaml_dict[cfg.Vedio_backbone]["config"]
            
        self.Vedio_block = uniformerv2_b16(
           t_size=num_segments,
           num_classes=num_class,
           **self.Vedio_block_cfg
        )
        #! 融合多焦段特征
        # self.pos_emb = PositionalEncoding(1024,0.8)
        # self.Vedion_emb_att = Attention(self.Vedio_block_cfg["n_dim"],num_head=4)
        self.fusion_block = Fusion_block(focal_cnt=5)
        
        # self.feature_dropout = nn.Dropout(0.5)
        # 临床数据embedding部分 
        #! 输入维度
        self.emb_inDim = 32
        self.emb_outDim = 32
        self.embedding = EggsDataNet(self.emb_inDim, self.emb_outDim)

        # mutan多模态融合部分 
        self.mutan_data_inDim = 32
        self.mutan_vedio_inDim = self.Vedio_block_cfg["n_dim"]
        self.mutan_outDim = 32
        self.mutan_layers=5
        #! 修改了输入维度，因为data和Uniformer抽取的vedio编码长度不一致的情况
        self.mutan = MutanFusion(self.mutan_vedio_inDim, self.mutan_data_inDim, self.mutan_outDim, self.mutan_layers)
        # mutan的输入是 bi-LSTM的输出向量 以及临床数据的embedding向量 输出与这些向量维度一样的向量 

        self.final_fc = nn.Linear(self.mutan_outDim, num_class)
        self.dropout_layer = nn.Dropout(p=0.5)

        '''添加部分到此结束'''
        
        self.load_pretrain_vedio_block()
        self.freeze_Vedio_block()
        self.prepare_focal_opt()
        if cfg.use_ckpt:
            self.prepare_from_ckpt(cfg.ckpt_file)
        
    def freeze_Vedio_block(self):
        logger.info("freeze Vedio block")
        for p in self.Vedio_block.parameters():
            p.requires_grad = False
        for p in self.fusion_block.backbone.res_block1.parameters():
            p.requires_grad = False
            
    def unfreeze_Vedio_block(self):
        logger.info("unfreeze Vedio block")
        for p in self.Vedio_block.parameters():
            p.requires_grad = True

    def set_finetune_params(self):
        name_prefix = ["bias"]
        for name,p in self.Vedio_block.named_parameters():
            full_name = "Vedio_block." + name
            # if "resblocks" in name: 
            #     if int(full_name.split('.')[3])>22:
            #         p.requires_grad = True
            #         logger.info("finetune param: " + full_name)
            #         self.finetune_param_name_lst.append(full_name)
            flag = 0
            for i in name_prefix:
                if i in full_name:
                    flag = 1
            if flag:
                p.requires_grad = True
                logger.info("finetune param: " + full_name)
                self.finetune_param_name_lst.append(full_name)
    
    def prepare_from_ckpt(self,ckpt_file):
        # ckpt_file = "/data2/liangzhijia/Blastocyst/Uniformerv2_ART/run/train_16/weight/best_auc.pth"
        state_dict = torch.load(ckpt_file,map_location='cpu')
        logger.info("load ckpt " + ckpt_file + " strict = True")
        self.load_state_dict(state_dict,strict=True)
    
    def prepare_focal_opt(self):
        self.Vedio_block.conv1 = nn.Conv3d(64, self.Vedio_block_cfg["n_dim"], (3, 4, 4), (2, 4, 4), (1, 0, 0), bias=False).requires_grad_()
        # self.Vedio_block.conv1 = nn.Conv3d(7, self.Vedio_block_cfg["n_dim"], (3, 16, 16), (2, 16, 16), (1, 0, 0), bias=False).requires_grad_()
        logger.info("change uniformer.conv1 adapt to fusion block")
    
    #! 用Uniformer论文中训练好的权重初始化视频模块
    def load_pretrain_vedio_block(self):
        pretrain_file = self.Vedio_block_ckpt_file
        logger.info("use pretrain weight " + pretrain_file)
        state_dict = torch.load(pretrain_file)
        # with open("pretrain_uni_l14.txt",'w') as f:
        #     for key in state_dict.keys():
        #         f.write(key + "    " + str(state_dict[key].shape) + "\n")
        # with open("uni_l14.txt",'w') as f:
        #     for key in self.Vedio_block.state_dict().keys():
        #         f.write(key + "    " + str(self.Vedio_block.state_dict()[key].shape) + "\n")
        this_state_dict = self.Vedio_block.state_dict()
        init_pretrain_state_dict = OrderedDict([
                (key[9:], value) for key,value in state_dict.items()
            ]
        )
        not_load_layers = [
                k
                for k in init_pretrain_state_dict.keys()
                if k not in this_state_dict.keys()
            ]
        if not_load_layers and torch.cuda.current_device() == 0:
            for k in not_load_layers:
                # print("Network weights {} not loaded.".format(k))
                logger.info("Network weights {} not loaded.".format(k))
        self.pretrain_param_name_lst = [
                k
                for k in init_pretrain_state_dict.keys()
                if k in this_state_dict.keys()
            ]
        pretrain_state_dict = OrderedDict([
                (key, value) for key,value in init_pretrain_state_dict.items() if key not in not_load_layers
            ]   
        )
        self.Vedio_block.load_state_dict(pretrain_state_dict,strict=True)
        logger.info("load pretrain weight " + pretrain_file + " strict = True")

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

    def partialBN(self, enable):
        self._enable_pbn = enable
    
    def init_param(self):
        for name, p in self.named_parameters():
            if "lmhra2.pos_embed.3" in name: #Local UniBlock输出的linear层
                nn.init.zeros_(p)
            if "temporal_cls_token" in name: #可学query
                nn.init.zeros_(p)
            if "c_proj" in name and "dec" in name: #Global UniBlock中FFN的输出linear
                nn.init.zeros_(p)
            if "balance" in name: #结尾加权
                nn.init.zeros_(p)
    
    def get_optim_policies(self):
        # self.set_finetune_params()
        
        finetune_vedio_param_lst = []
        rest_vedio_param_lst = []
        # Vedion_emb_att_lst = []
        emb_param_lst = []
        mutan_param_lst = []
        other_param_lst = []
        
        for name, m in self.named_modules():
            if "Vedio_block" in name:
                for sup_name,p in m.named_parameters(recurse=False):
                    full_name = name + "." + sup_name
                    if p.requires_grad:
                        if full_name in self.finetune_param_name_lst:
                            finetune_vedio_param_lst.append(p)
                        else:
                            rest_vedio_param_lst.append(p)
            # elif "Vedion_emb_att" in name:
            #     for name,p in m.named_parameters(recurse=False):
            #         Vedion_emb_att_lst.append(p)
            elif "embedding" in name:
                for name,p in m.named_parameters(recurse=False):
                    emb_param_lst.append(p)
            elif "mutan" in name:
                for name,p in m.named_parameters(recurse=False):
                    mutan_param_lst.append(p)
            else:
                for name,p in m.named_parameters(recurse=False):
                    if p.requires_grad:
                        other_param_lst.append(p)
            
        
        return [
            # {"params": finetune_vedio_param_lst, "lr_mult": 1, "name": "finetune_param"},
            {"params": rest_vedio_param_lst, "lr_mult": 1, "name": "rest_vedio_param"},
            # {"params": Vedion_emb_att_lst, "lr_mult": 1, "name": "Vedion_emb_att"},
            {"params": emb_param_lst, "lr_mult": 1, "name": "emb_param"},
            {"params": mutan_param_lst, "lr_mult": 1, "name": "mutan_param"},
            {"params": other_param_lst, "lr_mult": 1, "name": "other_param"}
        ]

    def forward(self, input, emb_index, femaleAge, maleAge):
        input = self.fusion_block(input)
        base_out = self.Vedio_block(input)
        # base_out = []
        # b, c, fo, frame_cnt, h, w = input.shape
        # for i in range(fo):
        #     base_out.append(self.Vedio_block(input[:,:,i,:,:]).unsqueeze(1))
        # main_video_output = base_out[0][:,0,:]
        # base_out = torch.cat(base_out,dim=1)
        # base_out = self.pos_emb(base_out)
        # video_output = self.Vedion_emb_att(base_out) + main_video_output
        #! 修改, 将TSM换成Uniformer
        video_output = base_out
        # video_output = self.feature_dropout(base_out)

        
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
        
    
