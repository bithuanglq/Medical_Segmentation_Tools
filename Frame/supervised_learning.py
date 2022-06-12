import os
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
import random
from tqdm import tqdm
import wandb
from collections import Counter
import json

from dataloader import *
from metirc import *
from unet import *





# Hyper parameters
BATCH_SIZE = 4
LR_INITIAL = 1e-6
EPOCH = 30
use_wandb = False   
DATAPARALLEL = False
main_gpu = 0

if use_wandb:
    # 初始化一个wandb run, 并设置超参数
    # Initialize a new run
    experiment = wandb.init(project="BraTS",entity='hlq',save_code=True) 

    # config is a variable that holds and saves hyper parameters and inputs
    experiment.config.update(dict(epochs=EPOCH, batch_size=BATCH_SIZE, lr_initial=LR_INITIAL))


work_path = '/home/hlq/Project/BraTS'
data_path = './BraTS2018/Data_Training/data.json'
weight_path = './checkpoint/model'
txt_path = './checkpoint/val.txt'
pth_path = None


os.chdir(work_path)
print(os.getcwd())


# GPU CUDA Parallel
if DATAPARALLEL:
    GPUs = [4,5,6,7]
    # GPUs = [0,1,2,3]
    main_gpu = GPUs[0]

torch.cuda.set_device('cuda:{}'.format(main_gpu))
device = torch.device('cuda:{}'.format(main_gpu) if torch.cuda.is_available() else 'cpu')
print("device=",device)
torch.backends.cudnn.benchmark = True   # 加快训练














def set_random_seed(seed=10):
    random.seed(seed)                  # random 模块中的随机种子，random是python中用于产生随机数的模块
    np.random.seed(seed)               # numpy中的随机种子
    torch.manual_seed(seed)            # 为CPU设置随机种子
    torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True   # deterministic用来固定内部随机性，即每次开机训练输入相同则输出大致相同


def freeze_but_bn(m:torch.nn.Module):
    classname = m.__class__.__name__
    if classname  not in ['BatchNorm2d','ReLU','Conv2d']:     
        for mm in m.children():      # 不能用 .modules()
            freeze_but_bn(mm)
        return
    freeze_but_bn.count += 1
    # print(classname,'第',freeze_but_bn.count,'层')
    if classname.find('BatchNorm')!=-1:
        for params in m.parameters():
            params.requires_grad = True
    else:
        for params in m.parameters():
            params.requires_grad = False
freeze_but_bn.count = 0


if __name__=='__main__':
    set_random_seed(10) # 设置随机数种子

    train_dataloader = DataLoader(MyDataset(data_train,label_train),batch_size=BATCH_SIZE,shuffle=True,num_workers=16,pin_memory=True,prefetch_factor=4)

    val_dataloader = DataLoader(MyDataset(data_val,label_val),batch_size=BATCH_SIZE,shuffle=True,num_workers=16,pin_memory=True,prefetch_factor=4)
    print('Load data Successfully!')


    net = UNet()
    if DATAPARALLEL:
        net = nn.DataParallel(net.to(device), device_ids=GPUs,output_device=GPUs[0])
    else:
        net = net.to(device)
    if pth_path:
        net.load_state_dict(torch.load(pth_path))
    # net.apply(freeze_but_bn)  # freeze layers but BN



    optimizer = torch.optim.Adam(net.parameters(),lr=LR_INITIAL)
    scheduler  = optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.1)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',patience=2)
    # lr_lst = [1e-4,1e-4,...]
    # scheduler = MultiplicativeLR(optimizer, lr_lambda=lambda epoch: lr_lst[epoch])
    loss_func = nn.BCELoss()


    # # 混合精度
    scaler = GradScaler()

    if use_wandb:
        wandb.watch(net,log='all')
    with open(txt_path,'w') as f:
        f.write('Start....\n')



    for epoch in range(EPOCH):
        # training
        with tqdm(train_dataloader,desc=f'Training--{epoch}',unit='img',unit_scale=True) as pbar:
            pbar.write(f'--------------开始第{epoch}轮！-------------')
            train_loss_per_epoch,len_train = 0,0
            for i,(image, _, mask,_) in enumerate(pbar):
                net.train()

                image,mask = image.float().cuda(non_blocking=True), mask.float().cuda(non_blocking=True)
                # image: (N,3,256,256) torch float
                # mask: (N,n,256,256) torch float


                optimizer.zero_grad()
                with autocast():
                    out = net(image)    # (N,n,256,256) torch float
                    train_loss = loss_func(out, mask)

                scaler.scale(train_loss).backward()
                scaler.step(optimizer)
                scaler.update()




                train_loss_per_epoch += train_loss.item()
                len_train += 1

            if use_wandb:
                experiment.log({
                'train loss': train_loss_per_epoch/(len_train+1e-1),
                'epoch': epoch
                })
            with open(txt_path,'a') as f:
                f.write(f'\n\nepoch:{epoch}\tloss:{train_loss_per_epoch/(len_train+1e-1)}\n')
            # 保存参数
            if True:
                torch.save(net.state_dict(), f'{weight_path}/ckpt_best_{epoch}.pth')
                pbar.write(f'--------------模型 ckpt_best_{epoch}.pth 保存完成！-------------')
    
        if True:
            # validation and save the params
            with tqdm(val_dataloader,desc=f'Validation--{epoch}',unit='img',unit_scale=True) as pbar2:
                pbar2.write(f'----开始验证：-----')
                net.eval()
                val_acc,recall,precision = 0,0,0
                with torch.no_grad():
                    length = 0
                    dice = 0
                    for i,(val_image, _, val_mask,_) in enumerate(pbar2):
                        val_image,val_mask = val_image.float().cuda(non_blocking=True), val_mask.float().cuda(non_blocking=True)
                        val_out = net(val_image)
                        dice += dice_coef_loss(input=val_out, target=val_mask)
                        length += 1
                    length += 1e-1
                    dice = dice/length
                    with open(txt_path,'a') as f:
                        f.write(f'Dice:{dice}\n')
                        lr = optimizer.param_groups[0]['lr']
                        f.write(f'learning rate:{lr}\n')

                    if use_wandb:
                        wandb.save(f'{weight_path}/ckpt_best_{epoch}.pth') 
                        experiment.log({
                                        'learning rate': optimizer.param_groups[0]['lr'],
                                        'dice': dice.item()
                                    })

                    # lr scheduler
                    scheduler.step()



