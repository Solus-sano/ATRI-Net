

"""统计数据中的正负样本比"""

# file_lst = ["train_data.txt", "val_data.txt", "test_data.txt"]
file_lst = ["train_data_clean.txt","test_data_clean.txt"]


for file_name in file_lst:

    with open(file_name,'r') as f:
        a = 0; b = 0
        for item in f.readlines():
            tmp = int(item.split(' ')[-5])
            if tmp: a+=1
            else: b+=1
        print(file_name, "1: ",a," 0: ",b," rate: ",b/(a+b))