import pandas as pd
import numpy as np
import chardet
import os
from tqdm import tqdm
"""
处理“timelapse_第四次合并整理后1.xlsx”中数据, 并写入txt中
txt data example: /data2/lizhida/datasets/data3/2021/869/191/D2021.10.19_S00191_I0869_D_WELL07 965 0 30 29 0 1
即一行中的元素为: 视频路径、帧数、非整倍体_完全、年龄女、年龄男、不良孕史、是否易位
"""

def check_file(file_path,frame_cnt):
    
    for root,dirs,files in os.walk(file_path):
        frame_lst = os.listdir(root)

        if len(frame_lst)<7: return 0 #检查焦段是否完整
        
        for dir in dirs:
            if not len(os.listdir(os.path.join(root,dir)))==frame_cnt: return 0   #检查各个焦段帧数是否完整
        return 1

def main():
    csv_path = "/data2/liangzhijia/Blastocyst/Uniformerv2_ART/data_opt/data2.xlsx"
    data = pd.read_excel(csv_path,engine='openpyxl',dtype=str).iloc[:,[1,50,51,52,64,2,3,5,62]].dropna(subset=["是否易位","非整倍体_完全"]).fillna(0)
    data.replace({'是':1,'否':0,'1(DMD)':1}, inplace=True)
    data = data.values
    print(data.shape)
    
    lst = []
    print('prepare total data: ')
    for item in tqdm(data):
        ip = None
        # example: D2020.02.24_S00145_I0868_D_WELL01
        if '/' in str(item[2]): continue
        
        file_name = 'D'+item[0].replace('-','.')+'_S{:05d}_I{:04d}_D_WELL{:02d}'.format(int(item[2]),int(item[1]),int(item[[3]]))
        file_path = '/data2/lizhida/datasets/data3/' + item[0][:4] + '/' + str(int(item[1])) + '/' + str(int(item[2])) + '/' + file_name + '/' + 'F0'
        
        if not os.path.exists(file_path): continue
        frames_cnt = len(os.listdir(file_path))
        if check_file(file_path[:-3],frames_cnt)==0: continue #只取有完整焦段的数据
        
        # print(frames_cnt)
        label, femaleage, maleage, parental, rearrange = int(item[-5]), str(item[-4]), str(item[-3]), str(item[-2]), str(item[-1])
        ip = file_path[:-3] + ' ' + str(frames_cnt) + ' ' + str(label) + ' ' + femaleage + ' ' + maleage + ' ' + parental + ' ' + rearrange + '\n'

        lst.append(ip)
    
    total_len = len(lst)
    print("total data cnt: ",total_len)
    a1,a2 = int(total_len * 0.7), int(total_len * 0.85)
            
    with open('total_data.txt','w') as f:
        for i in lst:
            f.write(i)
    
    # with open('train_data.txt','w') as f:
    #     for i in lst[:a1]:
    #         f.write(i)
            
    # with open('val_data.txt','w') as f:
    #     for i in lst[a1:a2]:
    #         f.write(i)
        
    # with open('test_data.txt','w') as f:
    #     for i in lst[a2:]:
    #         f.write(i)


    
if __name__ == '__main__':
    main()
    
    