import os
import time
import shutil
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
import torchvision
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_curve, auc

from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from opts import parser
from ops import dataset_config
from ops.utils import AverageMeter, accuracy
from ops.temporal_shift import make_temporal_pool

from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()
    args.gpu_ids = [str(i) for i in args.gpu_ids]
    gpu_ids = ', '.join(args.gpu_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    print("---- 使用整倍体识别模型预训练 24 ----")
    # 没有增加新数据的 
    # num_class, args.train_list, args.val_list, args.root_path, prefix = 2 , r'/home/xiexiang/MICCAI_code/ypx/ZBT/exchange_data/video_data_exchange/train_test/seed4/train.txt', r'/home/xiexiang/MICCAI_code/ypx/ZBT/exchange_data/video_data_exchange/train_test/seed4/test.txt', '' , 'img_{:05d}.jpg'
    # data_path = r'/home/xiexiang/MICCAI_code/ypx/ZBT/pure_data_annotation/total.xlsx'
    # 增加了新数据的 
    # num_class, args.train_list, args.val_list, args.root_path, prefix = 2 , r'/home/xiexiang/MICCAI_code/ypx/ZBT/重新跑的实验/annotation/add_new_annotation/train.txt', r'/home/xiexiang/MICCAI_code/ypx/ZBT/重新跑的实验/annotation/add_new_annotation/test.txt', '' , 'img_{:05d}.jpg'
    # data_path = r'/home/xiexiang/MICCAI_code/ypx/ZBT/重新跑的实验/annotation/add_new_annotation/total.xlsx'

    # 新数据交换的 
    # num_class, args.train_list, args.val_list, args.root_path, prefix = 2 , r'/home/xiexiang/MICCAI_code/ypx/ZBT/重新跑的实验/annotation/exchange_new_data/seed6/train.txt', r'/home/xiexiang/MICCAI_code/ypx/ZBT/重新跑的实验/annotation/exchange_new_data/seed6/test.txt', '' , 'img_{:05d}.jpg'
    # data_path = r'/home/xiexiang/MICCAI_code/ypx/ZBT/重新跑的实验/annotation/add_new_annotation/total.xlsx'

    # 矫正的新数据 下面的是把train和val一块拿来训练 
    # num_class, args.train_list, args.val_list, args.root_path, prefix = 2 , r'/home/xiexiang/MICCAI_code/ypx/ZBT/重新跑的实验/annotation/correct_new_annotation/train.txt', r'/home/xiexiang/MICCAI_code/ypx/ZBT/重新跑的实验/annotation/correct_new_annotation/test.txt', '' , 'img_{:05d}.jpg'
    # num_class, args.train_list, args.val_list, args.root_path, prefix = 2 , r'/home/xiexiang/MICCAI_code/ypx/ZBT/重新跑的实验/annotation/correct_new_annotation/train_and_val.txt', r'/home/xiexiang/MICCAI_code/ypx/ZBT/重新跑的实验/annotation/correct_new_annotation/test.txt', '' , 'img_{:05d}.jpg'
    # data_path = r'/home/xiexiang/MICCAI_code/ypx/ZBT/重新跑的实验/annotation/add_new_annotation/total.xlsx'


    # 重新计算了256的 
    # num_class, args.train_list, args.val_list, args.root_path, prefix = 2 , r'/home/xiexiang/MICCAI_code/ypx/ZBT/重新跑的实验/annotation/新的256数据annotation/train.txt', r'/home/xiexiang/MICCAI_code/ypx/ZBT/重新跑的实验/annotation/新的256数据annotation/test.txt', '' , 'img_{:05d}.jpg'
    # data_path = r'/home/xiexiang/MICCAI_code/ypx/ZBT/重新跑的实验/annotation/add_new_annotation/total.xlsx'
    # args.test_list = r'/home/xiexiang/MICCAI_code/ypx/ZBT/重新跑的实验/annotation/新的256数据annotation/val.txt'

    # 重新计算了256的 交换了新数据的train和val
    num_class, args.train_list, args.val_list, args.root_path, prefix = 2 , r'/home/xiexiang/MICCAI_code/ypx/ZBT/重新跑的实验/annotation/新的256数据annotation/exchange_newData/train_24.txt', r'/home/xiexiang/MICCAI_code/ypx/ZBT/重新跑的实验/annotation/新的256数据annotation/exchange_newData/test.txt', '' , 'img_{:05d}.jpg'
    data_path = r'/home/xiexiang/MICCAI_code/ypx/ZBT/重新跑的实验/annotation/add_new_annotation/total.xlsx'
    args.test_list = r'/home/xiexiang/MICCAI_code/ypx/ZBT/重新跑的实验/annotation/新的256数据annotation/exchange_newData/val_24.txt'
    
    full_arch_name = '3block_NL_Gaussian_LSTM'
    args.store_name = '_'.join(
        [full_arch_name, args.consensus_type, 'segment%d' % args.num_segments,'e{}'.format(args.epochs)])

    # args.store_name += '_newData_pretrain_v4'
    # args.store_name += '_重新计算的256_v1'
    args.store_name += '_整倍体预训练(无NL)24_交换新数据中的train和val_v20'
    #创建存储模型的文件以及点日志文件
    check_rootfolders()

    model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn,
                pretrain=args.pretrain,
                is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place,
                fc_lr5=True,
                temporal_pool=args.temporal_pool,
                non_local=args.non_local)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()

    train_augmentation = model.get_augmentation(flip=False if 'something' in args.dataset or 'jester' in args.dataset else True)

    model = torch.nn.DataParallel(model, device_ids=range(len(args.gpu_ids))).cuda()

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    #bu guan
    if args.resume:
        if args.temporal_pool:  # early temporal pool so that we can load the state_dict
            make_temporal_pool(model.module.base_model, args.num_segments)
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    #bu guan
    if args.tune_from:
        # 囊胚预测的模型 
        # pretrain_model_path_1 = r'/home/xiexiang/MICCAI_code/ypx/xiexiang/eggs_TSM_CWA_record/checkpoint_Gray_3block/resnet50_Gray_3block_8_blockres_RGB_identity_segment28_e64_kinetic_gongxiang_3focal/24.pth'
        # 整倍体识别模型 
        pretrain_model_path_1 = r'/home/xiexiang/MICCAI_code/ypx/ZBT/record/models/3block_with_data/exchange_train_test/Patient_Data_TSM_Gray_3block_identity_segment56_e256_exchange_seed4_train_test_v3/24.pth'
        # 用于修正版 
        # pretrain_model_path_1 = r'/home/xiexiang/MICCAI_code/ypx/ZBT/重新跑的实验/record/models/3block_NL_Gaussian/new_Data_TSM_Gray_3block_NL_Gaussian_identity_segment28_e256_newData_pretrain_v3/12.pth'
        
        # 3Focal pretrain
        # pretrain_model_path_1 = r'/home/xiexiang/MICCAI_code/ypx/ZBT/重新跑的实验/record/models/3block_NL_Gaussian/new_Data_TSM_Gray_3block_NL_Gaussian_identity_segment28_e256_correct_v1/13.pth'
        
        print('loading pretrain model:', pretrain_model_path_1)
        pretrain_model = torch.load(pretrain_model_path_1)
        sd = pretrain_model.state_dict()
        model_dict = model.state_dict()
        # 囊胚预测的模型 
        # sd = {k: v for k, v in sd.items() if 'fc' not in k}
        # 整倍体预测的模型
        sd = {k: v for k, v in sd.items()}

        model_dict.update(sd)
        model.load_state_dict(model_dict)

        for name, parameters in pretrain_model.named_parameters():
            if 'fc' in name or 'mlp' in name or 'down_dim' in name:
                continue
            if torch.equal(parameters.cpu(), model.state_dict()[name].cpu()):
                continue
            else:
                print(name)
                assert False

        print('......finished pretrain initialization......')

    if args.temporal_pool and not args.resume:
        make_temporal_pool(model.module.base_model, args.num_segments)

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
        print('mean ',input_mean,' std ',input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    train_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.train_list, table_path=data_path, num_segments=args.num_segments,
                   new_length=data_length,
                   modality='Gray',
                   image_tmpl=prefix,
                   transform=torchvision.transforms.Compose([
                       # GroupScale(int(scale_size)),
                       train_augmentation,
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.val_list, table_path=data_path, num_segments=args.num_segments,
                   new_length=data_length,
                   modality='Gray',
                   image_tmpl=prefix,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                    #    GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.test_list, table_path=data_path, num_segments=args.num_segments,
                   new_length=data_length,
                   modality='Gray',
                   image_tmpl=prefix,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                    #    GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    '''训练和验证'''
    log_training = open(os.path.join(args.root_log, args.store_name, 'log.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    best_epoch = 0
    best_roc_auc = 0
    best_roc_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, log_training)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            test_prec1, test_auc = test(test_loader, model, criterion, epoch, log_training, tf_writer)
            prec1, roc_auc = validate(val_loader, model, criterion, epoch, log_training, tf_writer)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            if is_best:
                best_epoch = epoch
            best_prec1 = max(prec1, best_prec1)

            if roc_auc> best_roc_auc:
                best_roc_auc = roc_auc
                best_roc_epoch = epoch 

            output_best = 'Best ACC: %.5f' % (best_prec1) + ' at val-'+str(int((best_epoch+1)/2))
            output_best += ' Best AUC: %.5f' % (best_roc_auc) + ' at val-'+str(int((best_roc_epoch+1)/2)) + '\n'
            print(output_best)
            log_training.write(output_best + '\n')
            log_training.flush()

            torch.save(model, args.root_model + '/' + args.store_name + '/' + str(int((epoch+1)/args.eval_freq))+'.pth')

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()   

    end = time.time()

    for i, (input, target, num_pos, embIndex, femaleAge, maleAge) in enumerate(train_loader):
        # print(femaleAge)
        # print(input.size())
        # print(target.size())
        # print(embIndex.size())
        # print(femaleAge.size())
        # print(maleAge.size())
        # print(target)
        # assert False

        batch_size = input.size(0)

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        embIndex_var = torch.autograd.Variable(embIndex)
        femaleAge_var = torch.autograd.Variable(femaleAge)
        maleAge_var = torch.autograd.Variable(maleAge)

        # compute output
        output = model(input_var, embIndex_var, femaleAge_var, maleAge_var)
        # print("输出的size：",input_var.size())
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target_var)
        # print(prec1)
        losses.update(loss.item(), batch_size)
        top1.update(prec1.item(), batch_size)

        # compute gradient and do SGD step
        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i+1 == len(train_loader):
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'ACC-{top1.avg:.3f}\t'.format(
                epoch, i+1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses,top1=top1,
                lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
            print(output)


def validate(val_loader, model, criterion, epoch, log=None, tf_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    # switch to evaluate mode
    model.eval()

    final_roc_label = []
    final_roc_pred = []

    end = time.time()
    with torch.no_grad():
        for i, (input, target, num_pos, embIndex, femaleAge, maleAge) in enumerate(val_loader):

            final_roc_label.extend(target.tolist())
          
            batch_size = input.size(0)
            target = target.cuda()

            # compute output
            output = model(input, embIndex, femaleAge, maleAge)

            loss = criterion(output, target)
            losses.update(loss.item(), batch_size)

            prec1 = accuracy(output.data, target)
            top1.update(prec1.item(), batch_size)

            output_softmax = nn.Softmax(dim=-1)(output)
            probility_true = [t[1] for t in output_softmax.tolist()]
            final_roc_pred.extend(probility_true)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    fpr, tpr, th = roc_curve(np.array(final_roc_label), np.array(final_roc_pred), pos_label=1)
    roc_auc = auc(fpr, tpr)

    output = str(int((epoch + 1) / 2))
    output += (
        'Validating Results: ACC-{top1.avg:.3f} ROC_AUC-{roc_auc:.3f}'.format(top1=top1, roc_auc=roc_auc))
    print(output)
    log.write('----------------------------------------------\n')
    log.write(output + '\n')
    log.flush()

    return top1.avg, roc_auc 


def test(val_loader, model, criterion, epoch, log=None, tf_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    # switch to evaluate mode
    model.eval()

    final_roc_label = []
    final_roc_pred = []

    end = time.time()
    with torch.no_grad():
        for i, (input, target, num_pos, embIndex, femaleAge, maleAge) in enumerate(val_loader):

            final_roc_label.extend(target.tolist())
          
            batch_size = input.size(0)
            target = target.cuda()

            # compute output
            output = model(input, embIndex, femaleAge, maleAge)

            loss = criterion(output, target)
            losses.update(loss.item(), batch_size)

            prec1 = accuracy(output.data, target)
            top1.update(prec1.item(), batch_size)

            output_softmax = nn.Softmax(dim=-1)(output)
            probility_true = [t[1] for t in output_softmax.tolist()]
            final_roc_pred.extend(probility_true)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    fpr, tpr, th = roc_curve(np.array(final_roc_label), np.array(final_roc_pred), pos_label=1)
    roc_auc = auc(fpr, tpr)

    output = str(int((epoch + 1) / 2))
    output += (
        'testing Results: ACC-{top1.avg:.3f} ROC_AUC-{roc_auc:.3f}'.format(top1=top1, roc_auc=roc_auc))
    print(output)
    log.write(output + '\n')
    log.flush()

    return top1.avg, roc_auc 


def save_checkpoint(state, is_best):
    filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_type == 'step':
        decay = 0.5 ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']

def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


if __name__ == '__main__':
    main()