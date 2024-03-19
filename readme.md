# 介绍

ATRI-Net完整代码，根据多焦段胚胎发育视频以及男方年龄、女方年龄、是否易位四个临床数据输入判别发育完成的囊胚是否是整倍体。

模型中使用的Uniformerv2：（github：https://github.com/OpenGVLab/UniFormerV2/tree/main）

# 代码

代码结构说明：

data_opt中为数据处理筛选部分，其中data2.xlsx为医院整合的未经清洗的初始数据，训练验证使用的是清洗后的test_data_clean.txt、train_data_clean.txt

dataset用于构建dataset和dataloader

extract_clip用于预处理clip的ViT模型(Uniformerv2初始化会用到）

models中为模型的构建

run中为训练、多卡运行所需代码

训练时调整run_config.py中的参数，然后运行python run/run_multi_node.py即可
