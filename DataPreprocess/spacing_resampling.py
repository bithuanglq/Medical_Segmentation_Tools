import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import PIL.Image as Image
from skimage.transform import resize
from batchgenerators.augmentations.utils import resize_segmentation
from scipy.ndimage.interpolation import map_coordinates
import torch
import torch.nn.functional as F



def show_certain_image_of_3d(datapath, slice):
    '''
        show slice image of nii

        :params datapath: the path of data file
        :params slice   : which slice you want to show
    '''   
    data = sitk.ReadImage(datapath)
    array = sitk.GetArrayFromImage(data)
    print(array.shape, np.unique(array))
    array2d = array[slice]
    # array2d[array2d>0] = 255.0                # if the nii is the seg file, then uncomment this
    print(np.unique(array2d), array2d.dtype)    # uint16 才行
    Image.fromarray(array2d).convert('RGB').save('show_certain_image_of_3d.png')   



target_spacing_percentile = 50  # median spacing
anisotropy_threshold = 3    # anisotropy



def compute_target_spacing(csvpath:str):
    '''
        e.g.    
                filename       sitk_shape                                        spacing
            0    amos_0001.nii.gz   (90, 636, 636)  (0.6882500052452087, 0.6882500052452087, 5.0)
            1    amos_0004.nii.gz   (78, 582, 582)  (0.6882500052452087, 0.6882500052452087, 5.0)
            2    amos_0005.nii.gz   (80, 635, 635)  (0.6882500052452087, 0.6882500052452087, 5.0)
            3    amos_0006.nii.gz   (99, 621, 621)  (0.6882500052452087, 0.6882500052452087, 5.0)
            4    amos_0007.nii.gz  (107, 578, 578)  (0.6882500052452087, 0.6882500052452087, 5.0)
            ..                ...              ...                                            ...
            195  amos_0404.nii.gz   (84, 597, 597)  (0.6882500052452087, 0.6882500052452087, 5.0)
            196  amos_0405.nii.gz   (90, 657, 657)  (0.6882500052452087, 0.6882500052452087, 5.0)
            197  amos_0406.nii.gz   (82, 620, 620)  (0.6882500052452087, 0.6882500052452087, 5.0)
            198  amos_0408.nii.gz  (104, 686, 686)  (0.6882500052452087, 0.6882500052452087, 5.0)
            199  amos_0410.nii.gz  (107, 651, 651)  (0.6882500052452087, 0.6882500052452087, 5.0)

            [200 rows x 3 columns]
    '''
    patient_nums = 200 


    df = pd.read_csv(csvpath)
    spacings = df['spacing'][:patient_nums]
    tmp_spacings = []
    for spacing in spacings:
        tmp_spacing = []
        for s in spacing[1:-1].split(','):
            tmp_spacing.append(float(s))
        tmp_spacings.append(tmp_spacing)
    spacings = np.array(tmp_spacings)   # (patient_nums, 3)

    target_spacing = np.percentile(np.vstack(spacings), target_spacing_percentile, 0)  # 取中位数
    print(target_spacing)    # [0.68825001 0.68825001 5.        ]



def resample_data(target_spacing:np.array, datapath:str, dirpath:str):
    '''
        resample data to the same spacing
        datapath: e.g. data/1.nii.gz
    '''

    # 根据spacing确定是否为 anisotropic
    do_separate_z = (np.max(target_spacing) / np.min(target_spacing)) > anisotropy_threshold    # True
    

    # resample
    if do_separate_z:
        data = sitk.ReadImage(os.path.join(datapath))
        array = sitk.GetArrayFromImage(data)
        array = np.transpose(array, (2,1,0))        # 维度与spacing维度是一一对应，之前array(z,y,x) spacing(x,y,z) size(x,y,z)
        dtype_data = array.dtype
        spacing = data.GetSpacing()
        shape = data.GetSize()
        new_shape = np.array([int(np.round(shape[i]*spacing[i]/target_spacing[i])) for i in range(len(shape))]) # 一定要取整

        # 先采样两个高分辨率（低spacing）轴
        axis = np.where(max(spacing) / np.array(spacing) == 1)[0]
        assert len(axis)==1, 'only one anisotropic axis supported'

        axis = axis[0]
        if axis == 0:
            new_shape_2d = new_shape[1:]
        elif axis == 1:
            new_shape_2d = new_shape[[0, 2]]
        else:
            new_shape_2d = new_shape[:-1]

        reshaped_data = []
        resize_fn = resize
        order = 3
        for slice_id in range(shape[axis]):
            if axis == 0:
                reshaped_data.append(resize_fn(array[slice_id], new_shape_2d, order).astype(dtype_data))
            elif axis == 1:
                reshaped_data.append(resize_fn(array[:, slice_id], new_shape_2d, order).astype(dtype_data))
            else:
                reshaped_data.append(resize_fn(array[:, :, slice_id], new_shape_2d, order).astype(dtype_data))
        reshaped_data = np.stack(reshaped_data, axis)



        # 再采样低分辨率（高spacing）轴
        if shape[axis] != new_shape[axis]:
            rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
            orig_rows, orig_cols, orig_dim = shape

            row_scale = float(orig_rows) / rows
            col_scale = float(orig_cols) / cols
            dim_scale = float(orig_dim) / dim

            map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]                 #  rows默认向上取整
            map_rows = row_scale * (map_rows + 0.5) - 0.5
            map_cols = col_scale * (map_cols + 0.5) - 0.5
            map_dims = dim_scale * (map_dims + 0.5) - 0.5

            coord_map = np.array([map_rows, map_cols, map_dims])
            reshaped_final_data = (map_coordinates(reshaped_data, coord_map, order=3, mode='nearest').astype(dtype_data))    # done with third order spline
        else:
            reshaped_final_data = reshaped_data
        reshaped_final_data = np.transpose(reshaped_final_data, (2,1,0))


        # save
        print(reshaped_final_data.shape)    # (z,y,x)
        resampled_data = sitk.GetImageFromArray(reshaped_final_data)
        resampled_data.SetSpacing(target_spacing)
        sitk.WriteImage(resampled_data, os.path.join(dirpath))



def resample_seg(target_spacing:np.array, origin, direction, datapath:str, dirpath:str):
    '''
        resample label to the same spacing
    '''

    # 根据spacing确定是否为 anisotropic
    do_separate_z = (np.max(target_spacing) / np.min(target_spacing)) > anisotropy_threshold    # True
    

    # resample
    if do_separate_z:
        data = sitk.ReadImage(os.path.join(datapath))
        array = sitk.GetArrayFromImage(data)
        array = np.transpose(array, (2,1,0))        # 维度与spacing维度是一一对应
        dtype_data = array.dtype
        spacing = data.GetSpacing()
        shape = data.GetSize()
        new_shape = np.array([int(np.round(shape[i]*spacing[i]/target_spacing[i])) for i in range(len(shape))])

        array = F.one_hot(torch.from_numpy(array.astype('int16')).type(torch.LongTensor))   # one hot 
        Array = array.numpy().astype(dtype_data)

        reshaped_final_data = []
        for c in range(Array.shape[-1]):
            array = Array[:,:,:,c]

            # 先采样两个高分辨率（低spacing）轴
            axis = np.where(max(spacing) / np.array(spacing) == 1)[0]
            assert len(axis)==1, 'only one anisotropic axis supported'

            axis = axis[0]
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            reshaped_data = []
            resize_fn = resize_segmentation
            order = 0
            for slice_id in range(shape[axis]):
                if axis == 0:
                    reshaped_data.append(resize_fn(array[slice_id], new_shape_2d, order).astype(dtype_data))
                elif axis == 1:
                    reshaped_data.append(resize_fn(array[:, slice_id], new_shape_2d, order).astype(dtype_data))
                else:
                    reshaped_data.append(resize_fn(array[:, :, slice_id], new_shape_2d, order).astype(dtype_data))
            reshaped_data = np.stack(reshaped_data, axis)



            # 再采样低分辨率（高spacing）轴
            if shape[axis] != new_shape[axis]:
                rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                orig_rows, orig_cols, orig_dim = shape

                row_scale = float(orig_rows) / rows
                col_scale = float(orig_cols) / cols
                dim_scale = float(orig_dim) / dim

                map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                map_rows = row_scale * (map_rows + 0.5) - 0.5
                map_cols = col_scale * (map_cols + 0.5) - 0.5
                map_dims = dim_scale * (map_dims + 0.5) - 0.5

                coord_map = np.array([map_rows, map_cols, map_dims])

                # interpolation
                unique_labels = np.unique(reshaped_data)
                reshaped = np.zeros(new_shape, dtype=dtype_data)

                for i, cl in enumerate(unique_labels):
                    # print(i,cl)
                    reshaped_multihot = np.round(
                        map_coordinates((reshaped_data == cl).astype(float), coord_map, order=1, mode='nearest'))   # linear interpolation
                    reshaped[reshaped_multihot > 0.5] = cl
                reshaped_final_data.append(reshaped.astype(dtype_data))
            else:
                reshaped_final_data.append(reshaped_data)

        # inverse one hot
        reshaped_final_data = np.array(reshaped_final_data, dtype=dtype_data)
        # print(reshaped_final_data.shape, reshaped_final_data.dtype) # (c, x, y, z)  dtype_data
        reshaped_final_data = np.argmax(reshaped_final_data, axis=0).astype('uint16')
        reshaped_final_data = np.transpose(reshaped_final_data, (2,1,0))


        # save
        print(reshaped_final_data.shape)
        resampled_data = sitk.GetImageFromArray(reshaped_final_data)
        resampled_data.SetSpacing(target_spacing)
        resampled_data.SetOrigin(origin)
        resampled_data.SetDirection(direction)
        sitk.WriteImage(resampled_data, os.path.join(dirpath))








'''
    ============================================================================
    Interpolation with a zero-order spline is nearest neighbor interpolation, 
    and interpolation with a first-order spline is linear interpolation.
    ============================================================================

>>> from scipy import ndimage
>>> a = np.arange(12.).reshape((4, 3))
>>> a
array([[ 0.,  1.,  2.],
       [ 3.,  4.,  5.],
       [ 6.,  7.,  8.],
       [ 9., 10., 11.]])
>>> ndimage.map_coordinates(a, [[0.5, 2], [0.5, 1]], order=1)
array([2., 7.])
>>> ndimage.map_coordinates(a, [[0.5, 2], [0.5, 1]], order=0)
array([4., 7.])
>>> ndimage.map_coordinates(a, [[0.5, 2], [0.5, 1]], order=0, mode='nearest')
array([4., 7.])
>>> ndimage.map_coordinates(a, [[0.5, 2], [0.5, 1]], order=1, mode='nearest')
array([2., 7.])
>>> ndimage.map_coordinates(a, [[0.5, 2], [0.5, 1]], order=1, mode='constant')
array([2., 7.])
>>> ndimage.map_coordinates(a, [[0.5, 2], [0.5, 1]], order=0, mode='constant')
array([4., 7.])
>>> ndimage.map_coordinates(a, [[0.5, 2], [0.5, 1]], order=3, mode='constant')
array([1.3625, 7.    ])
>>> ndimage.map_coordinates(a, [[0.5, 2], [0.5, 1]], order=3, mode='nearest')
array([1.67884758, 7.        ])
>>> ndimage.map_coordinates(a, [[0.5, 2], [0.5, 1]], order=3, mode='mirror')
array([1.3625, 7.    ])
>>> ndimage.map_coordinates(a, [[0.5, 2], [0.5, 1]], order=3, mode='reflect')
array([1.76411552, 7.        ])
>>> ndimage.map_coordinates(a, [[0.5, 2], [0.5, 1]], order=1, mode='reflect')
array([2., 7.])

'''
