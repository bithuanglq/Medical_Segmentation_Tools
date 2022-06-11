import os
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
import random

from dataloader import *
from metirc import dice_coef_loss

# Hyper parameters
BATCH_SIZE = 4
LR_INITIAL = 1e-3
EPOCH = 60
CLIP_VALUE = 0.05
main_gpu = 0


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



if __name__ == '__main__':
    set_random_seed(10) # 设置随机数种子
    train_set, val_set = split_dataset(data_path=data_path, per=0.7)
    train_dataloader = DataLoader(MyDataset(data_path, train_set),batch_size=BATCH_SIZE,shuffle=True,num_workers=16,pin_memory=True,prefetch_factor=4)
    val_dataloader = DataLoader(MyDataset(data_path, val_set),batch_size=BATCH_SIZE,shuffle=True,num_workers=16,pin_memory=True,prefetch_factor=4)
    print('Load data Successfully!')

    optimizer_G = optim.Adam(generator.parameters(), lr=LR_INITIAL)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR_INITIAL)


    for epoch in range(EPOCH):
        # train
        generator.train()
        for i, (image, mask) in enumerate(train_dataloader):

            image,mask = image.float().cuda(non_blocking=True), mask.float().cuda(non_blocking=True)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()
            fake_imgs = generator(image).detach()       # detach() 生成器不反向传播
            loss_D = -torch.mean(discriminator(image, mask)) + torch.mean(discriminator(fake_imgs, mask))
            loss_D.backward()
            optimizer_D.step()

            # clip parameters in D
            for p in discriminator.parameters():                # Clip weights of discriminator  裁剪分辨器D的权重是WGAN的特点，加速收敛
                p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)


            
            if epoch % 5 == 0:
                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()
                gen_imgs = generator(image)      # Generate a batch of images
                dice_loss = 1 - dice_coef_loss(gen_imgs, mask)
                ad_loss = -torch.mean(discriminator(gen_imgs, mask))           # Adversarial loss
                loss_G = dice_loss + ad_loss
                loss_G.backward()
                optimizer_G.step()
        

        # validation
        generator.eval()
        with torch.no_grad():
            for i, (image, mask) in enumerate(val_dataloader):

                image,mask = image.float().cuda(non_blocking=True), mask.float().cuda(non_blocking=True)

                output = generator(image)
                dice_score = dice_coef_loss(output, mask)

            print(dice_score.cpu().numpy())
