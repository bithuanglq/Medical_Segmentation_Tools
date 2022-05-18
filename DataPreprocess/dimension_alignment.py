import SimpleITK as sitk
import numpy as np



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