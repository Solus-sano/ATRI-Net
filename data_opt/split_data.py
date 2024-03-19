
lst = None

with open("total_data_clean.txt",'r') as f:
    lst = f.readlines()
    
a = int(len(lst)*0.8)

with open("train_data_clean.txt",'w') as f:
    for item in lst[:a]:
        f.write(item)
        
with open("test_data_clean.txt",'w') as f:
    for item in lst[a:]:
        f.write(item)
    