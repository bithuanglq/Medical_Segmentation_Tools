from torch.utils.data import Dataset
import numpy as np
import os



class MyDataset(Dataset):
    '''
        load data from filepath separately
    '''
    def __init__(self,datapath,labelpath) -> None:
        super().__init__()
        self.datapath = datapath
        self.labelpath = labelpath
    
    def __len__(self):
        return len(os.listdir(self.datapath))
    
    def __getitem__(self, index):
        image,mask = np.load(os.path.join(self.datapath, f'{index}.npy')), np.load(os.path.join(self.labelpath, f'{index}.npy'))
        # image:(3,256,256) numpy
        # mask: (9,256,256) numpy
        image = (image - image.mean())/(image.std())


        foreground,background = mask[:-1],np.expand_dims(mask[-1],axis=0)
        # 前景非0像素改为1，背景非1像素改为1
        foreground = np.sum(foreground,axis=0,keepdims=True)    # 合并类别进行粗分割

        foreground = np.clip(foreground,0,1)                                        # mask: (1,256,256)
        foreground_onehot = np.append(foreground,background,axis=0)          # mask_onehot: (2,256,256)
        return image,mask,foreground,foreground_onehot
