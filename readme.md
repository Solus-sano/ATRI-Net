# 介绍

处理的任务与之前ART中相同，即根据视频、男方年龄、女方年龄、是否易位四个输入判别发育完成的囊胚是否是整倍体。

模型中使用的Uniformerv2：（github：https://github.com/OpenGVLab/UniFormerV2/tree/main）

Uniformer具体使用的是UniFormerV2-L/14_16*224、UniFormerV2-L/14_32*224、UniFormerV2-B/16_16*224（权重在/data2/liangzhijia/ckpt下），sth-sthv2数据集预训练的效果更好。

Uniformerv2模型的特征输出维度是1024或768，修改了融合模块中视频输入处全连接层的输入维度。

目前主要做法有两种：

1）将ART中用于抽取视频特征的TSM+LSTM模块整个替换为Uniformerv2，其余不做改动，固定整个Uniformer模块进行训练，最高AUC=0.770，这种做法可以使用最大预训练模型L/14_32*224，但只能使用单焦段视频

2）在（1）的设计下，在Uniformer输入前加入焦段融合方法，AUC=0.800（5焦段），焦段融合一种是使用先前ART中通道注意力的做法，一种是借鉴Uniformer中LMHA模块的方法，3、5、7焦段下LMHA方法都要更好，但两种方法在7焦段下效果都下降了很多

# 代码

代码结构说明：

data_opt中为数据处理筛选部分，其中data2.xlsx为之前师兄整合的数据，test_data.txt、train_data.txt、val_data.txt为目前用于训练的数据

dataset用于构建dataset和dataloader，主要使用了ART中的代码

extract_clip用于预处理clip的ViT模型并保存到/data2/liangzhijia/ckpt下（Uniformerv2初始化会用到）

models中为模型的构建

run中为训练、多卡运行所需代码，以及保存的实验日志

训练时调整run_config.py中的参数，然后在/data2/liangzhijia/Blastocyst/Uniformerv2_ART下运行python run/run_multi_node.py即可
