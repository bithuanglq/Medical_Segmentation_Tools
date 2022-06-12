import numpy as np
import torch



def dice_loss_oneclass(input:torch.Tensor,target:torch.Tensor):
    # input:(N,x,y,z) torch
    # target:(N,x,y,z) torch
    inputs = input.flatten()
    targets = target.flatten()
    intersection = inputs*targets
    smooth = 1e-5
    dice_loss = 1 - (smooth+2*torch.sum(intersection))/(smooth+ torch.sum(inputs)+ torch.sum(targets))
    return dice_loss



def dice_coef_loss(input:torch.Tensor,target:torch.Tensor,alpha=[1,1,1]):   # len(alpha) == nclass
    # input: (N,nclass,x,y,z) torch 
    # target:(N,nclass,x,y,z) torch
    nclass = target.shape[1]
    alpha = torch.Tensor(alpha)
    dice_loss = torch.sum(torch.Tensor([alpha[i]*dice_loss_oneclass(input[:,i,:,:,:],target[:,i,:,:,:]) for i in range(nclass)]))
    dice_loss = dice_loss/torch.sum(alpha)
    return dice_loss
