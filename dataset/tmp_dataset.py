from os import path
import torch.utils.data as data
from PIL import Image
import os
import numpy as np
from numpy.random import randint
import pandas as pd 
import torchvision
from torchvision import transforms
import torch

import sys
sys.path.append(os.path.abspath("../"))
# from utils.transforms import GroupCenterCrop, ToTorchFormatTensor, Stack

def get_image_tmpl(path,focal):
    img_name = os.listdir(path+r'/'+focal)[0]
    return focal+'/'+img_name.split('RUN')[0]+'RUN'+'{}'+'.JPG'

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])
    #! 增加了临床数据的标签
    @property
    def femaleAge(self):
        return int(self._data[3])
    
    @property
    def maleAge(self):
        return int(self._data[4])
    
    @property
    def parental(self):
        return int(self._data[5])
    
    @property
    def rearrange(self):
        return int(self._data[6])

class Uniformer_DataSet(data.Dataset):
    def __init__(self, root_path, list_file,table_path=None,
                 num_segments=32, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None, group_transform=None,
                 random_shift=True, test_mode=False,
                 remove_missing=False, dense_sample=False, twice_sample=False):

        # self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.group_transform = group_transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.twice_sample = twice_sample  # twice sample for more validation
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if self.twice_sample:
            print('=> Using twice sample for the dataset...')

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()

        self.tabel = None
        if table_path:
            self.tablePath = table_path
            self.table = pd.read_excel(self.tablePath)
            self.table['S'] = self.table['S'].astype('str')
    #! 修改_load_image函数，因为数据集路径和原来不一样
    # def _load_image(self, directory, idx, focal):
    #     if self.modality == 'RGB' or self.modality == 'RGBDiff':
    #         try:
    #             return [Image.open(os.path.join(directory, focal, self.image_tmpl.format(idx))).convert('RGB')]
    #         except Exception:
    #             print('error loading image:',os.path.join(directory, focal, self.image_tmpl.format(idx)))
    #             return [Image.open(os.path.join(directory, focal, self.image_tmpl.format(1))).convert('RGB')]

    #     elif self.modality == 'Gray' :
    #         try:
    #             return [Image.open(os.path.join(directory, focal, self.image_tmpl.format(idx))).convert('L')]
    #         except Exception:
    #             print('error loading image:',os.path.join(directory, focal, self.image_tmpl.format(idx)))
    #             return [Image.open(os.path.join(directory, focal, self.image_tmpl.format(1))).convert('L')]

    #     #不用管
    #     else:
    #         assert False
    
    def _load_image(self, directory, idx, focal):
        embryoImageName = directory.split("/")[-1]
        img_path = os.path.join(directory, focal, embryoImageName+"_RUN"+str(idx).zfill(3))+(".JPG")
        # print(img_path)
        # assert os.path.exists(img_path)
        first_img_path = os.path.join(directory, focal, embryoImageName+"_RUN"+str(1).zfill(3))+(".JPG")
        if self.modality == 'RGB':
            try:
                return [Image.open(img_path).convert('RGB')]
            except Exception:
                print('error loading image:', img_path)
                return [Image.open(first_img_path).convert('RGB')]

        elif self.modality == 'Gray':
            try:
                return [Image.open(img_path).convert('L')]
            except Exception:
                print('error loading image:', img_path)
                return [Image.open(first_img_path).convert('L')]

    # def _get_patient_data(self, patient, well):
    #     try:
    #         index = self.table[(self.table['S']==patient) & (self.table['T']==well)].index.tolist()[0]
    #     except:
    #         index = self.table[(self.table['S']==patient.split('_')[-1]) & (self.table['T']==well)].index.tolist()[0]
    #         # print(patient, well)
            
    #     dataList = self.table.loc[[index]].values.tolist()[0]
        
    #     femaleMinAge = 20
    #     femaleMaxAge = 45
    #     maleMinAge = 20
    #     maleMaxAge = 50

    #     femaleAge = (dataList[0]-femaleMinAge)/(femaleMaxAge-femaleMinAge)
    #     maleAge = (dataList[1]-maleMinAge)/(maleMaxAge-maleMinAge)

    #     buYun = 2 if dataList[2]==0 else 3
    #     yiWei = 4 if dataList[3]==0 else 5
    #     data = np.array([0, 1, buYun, yiWei])
    #     return data, femaleAge, maleAge
    #! 使用新的get_patient_data函数，txt中包括了临床数据，不需要使用self.table
    def new_get_patient_data(self, record: VideoRecord):
        femaleAge = record.femaleAge
        maleAge = record.maleAge
        parental = record.parental
        rearrange = record.rearrange
        
        femaleMinAge = 20
        femaleMaxAge = 45
        maleMinAge = 20
        maleMaxAge = 50

        femaleAge = (femaleAge-femaleMinAge)/(femaleMaxAge-femaleMinAge)
        maleAge = (maleAge-maleMinAge)/(maleMaxAge-maleMinAge)

        buYun = 2 if parental==0 else 3
        yiWei = 4 if rearrange==0 else 5
        data = np.array([0, 1, buYun, yiWei])
        return data, femaleAge, maleAge

    def _parse_list(self):
        # check the frame number is large >3:
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        if not self.test_mode or self.remove_missing:
            tmp = [item for item in tmp if int(item[1]) >= 3]
        self.video_list = [VideoRecord(item) for item in tmp]

        #不管
        if self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            for v in self.video_list:
                v._data[1] = int(v._data[1]) / 2
        # print('video number:%d' % (len(self.video_list)))

    def _sample_indices(self, record):
        #不管
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1

        else:  # normal sample
            #获取拍摄图片的机器
            machine_id = record.path.split(r'/')[-1].split('_D')[0][-3:]
            if machine_id == '868':
                num_images_7days = 4*24*7
            elif machine_id =='869':
                num_images_7days = 6*24*7
            else:
                print('照片不是由868或869拍摄的')
                assert False

            # 获取它可以采样的帧数
            num_frames = min(num_images_7days, record.num_frames)
            num_segments = int(self.num_segments * num_frames / num_images_7days )

            # average_duration = (num_frames - self.new_length + 1) // num_segments
            average_duration = num_images_7days // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(num_segments)), average_duration) + randint(average_duration,size=num_segments)
            #不管
            elif record.num_frames > num_segments:
                offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=num_segments))
            else:
                offsets = np.zeros((num_segments,))
            return offsets + 1

    def _get_val_indices(self, record):
        #不管
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            #获取拍摄图片的机器
            machine_id = record.path.split(r'/')[-1].split('_D')[0][-3:]
            num_images_7days = 0
            if machine_id == '868':
                num_images_7days = 4*24*7
            elif machine_id =='869':
                num_images_7days = 6*24*7
            else:
                print('照片不是由868或869拍摄的')
                assert False

            num_frames = min(num_images_7days, record.num_frames)
            num_segments = int(self.num_segments * num_frames / num_images_7days)

            if num_frames > num_segments + self.new_length - 1:
                # tick = (num_frames - self.new_length + 1) / float(num_segments)
                tick = num_images_7days / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)])
            else:
                offsets = np.zeros((num_segments,))
            return offsets + 1

    def _get_test_indices(self, record):
        if self.dense_sample:
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        elif self.twice_sample:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)] +
                               [int(tick * x) for x in range(self.num_segments)])

            return offsets + 1
        else:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]
        # check this is a legit video folder
        #! 修改了视频路径
        file_name = record.path.split("/")[-1]+"_RUN"+str(1).zfill(3)+".JPG"
        full_path = os.path.join(record.path, 'F0', file_name)

        # if self.image_tmpl == 'flow_{}_{:05d}.jpg':
        #     file_name = self.image_tmpl.format('x', 1)
        #     full_path = os.path.join(self.root_path, record.path, file_name)
        # elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
        #     file_name = self.image_tmpl.format(int(record.path), 'x', 1)
        #     full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
        # else:
        #     file_name = self.image_tmpl.format(1)
        #     full_path = os.path.join(record.path,'F0', file_name)

        # while not os.path.exists(full_path):
        #     print('################## Not Found:', os.path.join(record.path, file_name))
        #     index = np.random.randint(len(self.video_list))
        #     record = self.video_list[index]
        #     if self.image_tmpl == 'flow_{}_{:05d}.jpg':
        #         file_name = self.image_tmpl.format('x', 1)
        #         full_path = os.path.join(self.root_path, record.path, file_name)
        #     elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
        #         file_name = self.image_tmpl.format(int(record.path), 'x', 1)
        #         full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
        #     else:
        #         file_name = self.image_tmpl.format(1)
        #         full_path = os.path.join(record.path, 'F0', file_name)

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices)
        # return self.get(record, segment_indices), index

    def get(self, record: VideoRecord, indices):

        focal_list1 = ['F-45','F-30','F-15','F0','F15','F30','F45']
        focal_list2 = ['F-75','F-50','F-25','F0','F25','F50','F75']

        num_pos = len(indices)
        padding_segments = self.num_segments - num_pos
        # indices = np.append(indices ,[1]*padding_segments)
        indices = np.append(indices ,[record.num_frames-1]*padding_segments)

        focal_list = os.listdir(record.path)
        if 'F-15' in focal_list:
            focal_list = focal_list1
        else:
            focal_list = focal_list2

        images = list()
        # for focal in focal_list:
        #     for seg_ind in indices:
        #         p = int(seg_ind)
        #         seg_imgs = self._load_image(record.path, p, focal)
        #         images.extend(seg_imgs)
        # focal = focal_list[0]
        for focal in focal_list:
            for seg_ind in indices:
                p = int(seg_ind)
                seg_imgs = self._load_image(record.path, p, focal)
                images.append(self.transform(seg_imgs[0]).unsqueeze(1))
        #! 增加针对视频的数据增强
        # if self.group_transform:
        #     images = self.group_transform(images)
        process_data = torch.cat(images,dim=1)
        process_data = process_data.reshape(3,len(focal_list),-1,process_data.size(-2),process_data.size(-1))

        '''这里添加了读取临床数据的代码'''
        #! txt中包含了临床数据，直接通过record读取数据，不使用self.table
        # if self.table:
        #     patient = str(record.path.split(r'/')[-2])
        #     well = int(record.path.split(r'/')[-1].split('WELL')[-1])
        #     embIndex, femaleAge, maleAge = self._get_patient_data(patient, well)
        # else:
        embIndex, femaleAge, maleAge = self.new_get_patient_data(record)
        '''到这里结束'''
        path_lst = [
            os.path.join('/data2/liangzhijia/datasets/embryo_feature','/'.join(record.path.split('/')[5:]),i) 
            for i in focal_list
        ]
        item = ' '.join(record._data[1:])
        # return process_data, record.label, num_pos
        return process_data, path_lst, item

    def __len__(self):
        return len(self.video_list)
    
    
def load_pretrain_vedio_block(net: torch.nn.Module,file):
    from collections import OrderedDict
    pretrain_file = file
    print("use pretrain weight " + pretrain_file)
    state_dict = torch.load(pretrain_file)
    # with open("pretrain_uni_l14.txt",'w') as f:
    #     for key in state_dict.keys():
    #         f.write(key + "    " + str(state_dict[key].shape) + "\n")
    # with open("uni_l14.txt",'w') as f:
    #     for key in self.Vedio_block.state_dict().keys():
    #         f.write(key + "    " + str(self.Vedio_block.state_dict()[key].shape) + "\n")
    this_state_dict = net.state_dict()
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
            print("Network weights {} not loaded.".format(k))
    pretrain_state_dict = OrderedDict([
            (key, value) for key,value in init_pretrain_state_dict.items() if key not in not_load_layers
        ]   
    )
    net.load_state_dict(pretrain_state_dict,strict=True)
    print("load pretrain weight " + pretrain_file + " strict = True")
    return net
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3,4,5,6,7'
    dataset = Uniformer_DataSet(
        "",
        "/data2/liangzhijia/Blastocyst/Uniformerv2_ART/data_opt/train_data_clean.txt",
        transform=torchvision.transforms.Compose([
                        transforms.ToTensor(),
                        transforms.CenterCrop(224)
                   ])
    )
    import yaml
    Vedio_block_cfg = None; Vedio_block_ckpt_file = None
    with open("/data2/liangzhijia/Blastocyst/Uniformerv2_ART/models/Uniformer_cfg.yaml",'r') as f:
        yaml_dict = yaml.safe_load(f)
        Vedio_block_ckpt_file = yaml_dict['sth2_l14_32x224']["ckpt"]
        Vedio_block_cfg = yaml_dict['sth2_l14_32x224']["config"]
    
    import sys
    sys.path.append(os.path.abspath('./'))
    from models.Uniformerv2 import uniformerv2_l14
    net = uniformerv2_l14(
        t_size=32,
        num_classes=2,
        **Vedio_block_cfg
    )
    net = load_pretrain_vedio_block(net,Vedio_block_ckpt_file).to('cuda')
    import tqdm

    with torch.no_grad():
        for imgs, path_lst, item in tqdm.tqdm(dataset):
            imgs = imgs.unsqueeze(0).cuda()
            for i in range(7):
                vedio = imgs[:,:,i,:,:,:]
                path = path_lst[i]
                pre_dir = '/'.join(path.split('/')[:-1])
                os.makedirs(pre_dir,exist_ok=True)

                fea = net(vedio)[0]
                torch.save(fea,path + '.pt')

    
    
