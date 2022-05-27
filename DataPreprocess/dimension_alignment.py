import os
import numpy as np
import SimpleITK as sitk






def resize_with_padding(datapath, dirpath, new_shape):
    '''
        resize with padding only in one dimension, e.g. (327, 512, 512) --> (368, 512, 512)  
        then 368 = 20(lb) + 327 + 21(ub),  ub = (368-327)//2 + (368-327)%2

        :params datapath : nii file path
        :params dirpath  : resized nii file path
        :params new_shape: padded shape
    '''

    data = sitk.ReadImage(datapath)
    array = sitk.GetArrayFromImage(data)
    spacing = data.GetSpacing()
    print(array.shape)

    if array.shape[0] > new_shape[0]:
        raise Exception("数据尺寸太大！")

    delta = new_shape[0] - array.shape[0]
    lb, ub = delta//2, delta//2 + delta%2
    if ub + lb != delta:
        raise Exception("区间划分错误！")

    array = np.pad(array, (
            (lb, ub),
            (0,0),
            (0,0)
    ), mode='edge')
    print(array.shape)

    # save
    padded_data = sitk.GetImageFromArray(array)
    padded_data.SetSpacing(spacing)
    sitk.WriteImage(padded_data, dirpath)





def resize_with_cropping(datapath, dirpath, new_shape):
    '''
        resize with cropping only in one dimension, e.g. (327, 512, 512) --> (256, 512, 512)  
        then 368 = -35(lb) + 327 - 36(ub),  ub = (327-256)//2 + (327-256)%2

        :params datapath : nii file path
        :params dirpath  : resized nii file path
        :params new_shape: cropped shape
    '''

    data = sitk.ReadImage(datapath)
    array = sitk.GetArrayFromImage(data)
    spacing = data.GetSpacing()
    print(array.shape)

    if array.shape[0] < new_shape[0]:
        raise Exception("数据尺寸太小！")
    
    delta =  array.shape[0] - new_shape[0]
    lb, ub = delta//2, delta//2 + delta%2
    if ub + lb != delta:
        raise Exception("区间划分错误！")

    array = array[lb:(array.shape[0] - ub), :, :]
    print(array.shape)

    # save
    padded_data = sitk.GetImageFromArray(array)
    padded_data.SetSpacing(spacing)
    sitk.WriteImage(padded_data, dirpath)
    
    
    
    
    
   

def padding_and_cropping(target_patchsize:np.array, datapath:str, dirpath:str):
    '''
        resize resampled  data, the same size as the input of NN
        datapath: e.g. data/1.nii.gz
        target_patchsize:   the pre-computed patchsize, as default, is the median  of datasets' shape for each axis.
    '''

    # If the patch size is not divisible by 2^n(d) for each axis, 
    # where n(d) is the number of downsampling operations, it is padded accordingly.
    mod = 2**4

    for i in range(len(target_patchsize)):
        slices = target_patchsize[i]
        if slices%mod > mod//2:
            target_patchsize[i] = (slices//mod+1)*mod
        else:
            target_patchsize[i] = (slices//mod)*mod
    print(target_patchsize)                             #  (z,y,x)  e.g. (96, 576, 576)


    data = sitk.ReadImage(os.path.join(datapath))
    array = sitk.GetArrayFromImage(data)
    spacing = data.GetSpacing()
    patchsize = array.shape
    patchsize = np.array(patchsize)

    # alignment of patchsize, padding before cropping
    delta = target_patchsize - patchsize
    delta = delta.astype('int64')
    lb,ub = delta//2, (delta//2)+(delta%2)

    array = np.pad(array, (
                (max(0, lb[0]), max(0, ub[0])),
                (max(0, lb[1]), max(0, ub[1])),
                (max(0, lb[2]), max(0, ub[2]))
        ), mode='edge')                             # padding mode

    array = array[
        max(0, -lb[0]) : (array.shape[0] - max(0, -ub[0])),
        max(0, -lb[1]) : (array.shape[1] - max(0, -ub[1])),
        max(0, -lb[2]) : (array.shape[2] - max(0, -ub[2]))
    ]




    # save
    print(patchsize, array.shape)
    padded_data = sitk.GetImageFromArray(array)
    padded_data.SetSpacing(spacing)
    sitk.WriteImage(padded_data, os.path.join(dirpath))
