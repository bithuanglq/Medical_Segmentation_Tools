import numpy as np
import torch
import torch.nn as nn



def dice_loss_oneclass(input:np.array,target:np.array):
    # input:(N,x,y,z) numpy
    # target:(N,x,y,z) numpy
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
    dice_loss = torch.sum(torch.Tensor([alpha[i]*dice_loss_oneclass(input[:,i,:,:],target[:,i,:,:]) for i in range(nclass)]))
    dice_loss = dice_loss/torch.sum(alpha)
    return dice_loss




class Dice_BCEWithLogits(nn.module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,input:torch.Tensor, target:torch.Tensor):
        # input: (N,3,x,y,z) torch 
        # target:(N,3,x,y,z) torch 0~1

        bcefunc = nn.BCEWithLogitsLoss()
        bceloss = bcefunc(input,target)
        input = nn.Sigmoid()(input)

        diceloss = dice_coef_loss(input,target) 

        return  bceloss + diceloss 


