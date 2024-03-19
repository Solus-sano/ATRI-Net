import pandas as pd
import numpy as np
import chardet
import os
from tqdm import tqdm

def check_file(file_path):
    
    for root,dirs,files in os.walk(file_path):
        focal_lst = os.listdir(root)
        if len(focal_lst) < 7: return 0
        
        frame_cnt = len(os.listdir(file_path + '/' + 'F0'))
        for dir in dirs:
            if not len(os.listdir(os.path.join(root,dir)))==frame_cnt: return 1   #检查各个焦段帧数是否完整
        return 2

def main():
    # csv_path = "/data2/liangzhijia/Blastocyst/ART/data_opt/data2.xlsx"
    # data = pd.read_excel(csv_path,engine='openpyxl',dtype=str).iloc[:,[1,50,51,52,64,2,3,5,62]].dropna(subset=["是否易位","非整倍体_完全"]).fillna(0)
    # data.replace({'是':1,'否':0,'1(DMD)':1}, inplace=True)
    # data = data.values
    
    csv_path = "/data2/liangzhijia/Blastocyst/AMSNet/data_opt/data2.xlsx"
    data = pd.read_excel(csv_path,dtype=str).iloc[:,[1,50,51,52,56]].dropna()
    data.replace({'是':1,'否':0}, inplace=True)
    data = data.values
    
    root_err_lst = []
    frame_cnt_err_lst = []
    focal_cnt_err_lst = []
    print('prepare total data: ')
    cnt=0
    for item in tqdm(data):
        ip = None
        # example: D2020.02.24_S00145_I0868_D_WELL01
        if '/' in str(item[2]):
            cnt+=1
            continue
        
        file_name = 'D'+item[0].replace('-','.')+'_S{:05d}_I{:04d}_D_WELL{:02d}'.format(int(item[2]),int(item[1]),int(item[[3]]))
        root_file_name = '/data2/lizhida/datasets/data3/' + item[0][:4] + '/' + str(int(item[1])) + '/' + str(int(item[2])) + '/' + file_name
        file_path = '/data2/lizhida/datasets/data3/' + item[0][:4] + '/' + str(int(item[1])) + '/' + str(int(item[2])) + '/' + file_name + '/' + 'F0'
        
        if not os.path.exists(root_file_name):
            root_err_lst.append(root_file_name)
            continue

        flag = check_file(file_path[:-3])
        if flag==0: #只取有完整焦段的数据
            focal_cnt_err_lst.append(root_file_name)
            continue
        
        elif flag==1:
            frame_cnt_err_lst.append(root_file_name)
            continue


        
        # print(frames_cnt)
        # label, femaleage, maleage, parental, rearrange = int(item[-5]), str(item[-4]), str(item[-3]), str(item[-2]), str(item[-1])
        # ip = file_path[:-3] + ' ' + str(frames_cnt) + ' ' + str(label) + ' ' + femaleage + ' ' + maleage + ' ' + parental + ' ' + rearrange + '\n'

        # lst.append(ip)
    root_err_len = len(root_err_lst)
    focal_err_len = len(focal_cnt_err_lst)
    frame_err_len = len(frame_cnt_err_lst)
    print("total data cnt: ",focal_err_len,' ',frame_err_len,' ',root_err_len)
    with open('root_err.txt','w') as f:
        for item in root_err_lst:
            f.write(item + '\n')
    
    with open('frame_cnt_err.txt','w') as f:
        for item in frame_cnt_err_lst:
            f.write(item + '\n')
            
    with open('focal_cnt_err.txt','w') as f:
        for item in focal_cnt_err_lst:
            f.write(item + '\n')
    print(cnt)
    
if __name__ == '__main__':
    main()
    # print(check_file('/data2/lizhida/datasets/data3/2020/868/157/D2020.05.26_S00157_I0868_D_WELL09',540))
    # csv_path = "/data2/liangzhijia/Blastocyst/AMSNet/data_opt/data2.xlsx"
    # data = pd.read_excel(csv_path).iloc[:,[1,50,51,52,56]].dropna()
    # data.replace({'是':1,'否':0}, inplace=True)
    # data = data.values
    # print(data[0][0])
    
    