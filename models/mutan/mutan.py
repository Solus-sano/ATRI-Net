import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class MutanFusion(nn.Module):
    def __init__(self, vedio_input_dim, data_input_dim, out_dim, num_layers=5):
        super(MutanFusion, self).__init__()
        # self.input_dim = input_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        hv = []
        for i in range(self.num_layers):
            do = nn.Dropout(p=0.5)
            lin = nn.Linear(vedio_input_dim, out_dim)

            hv.append(nn.Sequential(do, lin, nn.Tanh()))
        
        self.image_transformation_layers = nn.ModuleList(hv)
        
        hq = []
        for i in range(self.num_layers):
            do = nn.Dropout(p=0.5)
            lin = nn.Linear(data_input_dim, out_dim)
            hq.append(nn.Sequential(do, lin, nn.Tanh()))
        
        self.ques_transformation_layers = nn.ModuleList(hq)

    def forward(self, img_emb, ques_emb):
        
        batch_size = img_emb.size()[0]
        x_mm = []
        for i in range(self.num_layers):
            x_hv = img_emb
            x_hv = self.image_transformation_layers[i](x_hv)

            x_hq = ques_emb
            # print(x_hq.shape,' ')
            x_hq = self.ques_transformation_layers[i](x_hq)
            x_mm.append(torch.mul(x_hq, x_hv))
        
        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(batch_size, self.out_dim)
        x_mm = F.tanh(x_mm)
        return x_mm
