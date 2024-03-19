import random
random.seed(123)

"""平衡正负样本比"""

T_data_lst = []
F_data_lst = []

with open("total_data.txt",'r') as f:
    for item in f.readlines():
        flag = int(item.split(' ')[-5])
        if flag:
            T_data_lst.append(item)
        else:
            F_data_lst.append(item)
            
a = min(len(T_data_lst), len(F_data_lst))
total_data_lst = T_data_lst[:a] + F_data_lst[:a]

random.shuffle(total_data_lst)
with open("total_data_clean.txt","w") as f:
    for item in total_data_lst:
        f.write(item)
