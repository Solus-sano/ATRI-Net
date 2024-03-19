import os
import matplotlib.pyplot as plt
import torch

source = [
    # ["./test_record/test_38","CMBA-3focal"],
    ["./test_record/test_40","CMBA-5focal"],
    # ["test_41","CMBA-7focal"],
    # ["./train_record/train_48","LSHF-3focal"],
    ["./train_record/train_49","LSHF-5focal"],
    # ["test_45","LSHF-7focal"]
]

source_3focal = [
    ["./train_record/train_48","LSRA-3focal(AUC=0.785)"],
    ["./train_record/train_53","CBAM-3focal(AUC=0.775)"],
    ["./train_record/train_54","channel_cat-3focal(AUC=0.793)"],
    ["./train_record/train_59","Uniformerv2-3focal(AUC=0.730)"],
]

source_5focal = [
    ["./train_record/train_49","LSRA-5focal(AUC=0.800)"],
    ["./train_record/train_52","CBAM-5focal(AUC=0.798)"],
    ["./train_record/train_55","channel_cat-5focal(AUC=0.771)"],
    ["./train_record/train_60","Uniformerv2-5focal(AUC=0.716)"],
]

source_7focal = [
    ["./train_record/train_50","LSRA-7focal(AUC=0.787)"],
    ["./train_record/train_51","CBAM-7focal(AUC=0.742)"],
    ["./train_record/train_56","channel_cat-7focal(AUC=0.753)"],
    ["./train_record/train_61","Uniformerv2-7focal(AUC=0.747)"],
]

source_focals = [
    ["./train_record/train_48","ATRInet-3focal(AUC=0.785)"],
    ["./train_record/train_49","ATRInet-5focal(AUC=0.800)"],
    ["./train_record/train_50","ATRInet-7focal(AUC=0.787)"],
    ["./train_record/train_58","ATRInet-1focal(AUC=0.751)"],
    ["./train_record/train_57","Uniformerv2-1focal(AUC=0.703)"]
]

source_mutan = [
    ["./train_record/train_48","ATRInet-3focal(AUC=0.785)"],
    ["./train_record/train_49","ATRInet-5focal(AUC=0.800)"],
    ["./train_record/train_50","ATRInet-7focal(AUC=0.787)"],
    ["./train_record/train_59","Uniformerv2-3focal(AUC=0.730)"],
    ["./train_record/train_60","Uniformerv2-5focal(AUC=0.716)"],
    ["./train_record/train_61","Uniformerv2-7focal(AUC=0.747)"],
]

for item in source_7focal:
    pth_file = os.path.join(item[0],"weight/result.pth")
    if not os.path.exists(pth_file):
        pth_file = os.path.join(item[0],"weight/roc_result.pth")
    data = torch.load(pth_file)
    fpr, tpr = data["fpr"], data["tpr"]
    
    plt.plot(fpr, tpr, label=item[1])


# 为两条曲线添加文字标注
# plt.text(5, 0.5, 'sin(x)', fontsize=12, color='blue', ha='center')
# plt.text(5, -0.5, 'cos(x)', fontsize=12, color='orange', ha='center')

# 添加图例、标题和坐标轴标签
plt.legend()
plt.title('ROC CURVE')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('./figure/roc_7focal.png')