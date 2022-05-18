import SimpleITK as sitk
import PIL.Image as Image
import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage.interpolation import map_coordinates


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



def resample_seg(datapath, dirpath):
    '''
        3d resampling for seg, make that the voxel spacing of x,y,z axis is the same.
        Segmentation maps are resampled by converting them to one hot encodings.
        Each channel is then interpolated with linear interpolation and the segmentation mask is retrieved by an argmax operation.

        :params datapath: the path of the seg file      e.g. 'data/patient46_seg.nii.gz'
        :parmas dirpath : the stored path of resampled seg file 
    '''

    data = sitk.ReadImage(datapath)
    array = sitk.GetArrayFromImage(data)
    # print(array.shape)  # (817,512,512)
    # print(data.GetSize())   # (512,512,817)
    # print(data.GetSpacing()) # (0.625,0.625,0.25)
    dtype_data = array.dtype
    array = F.one_hot(torch.from_numpy(array.astype('int16')).type(torch.LongTensor))   # one hot 
    Array = array.numpy().astype(dtype_data)
    print(Array.shape, Array.dtype)             # (817,512,512) uint16


    reshaped_final_data = []
    for c in range(Array.shape[-1]):    # for each class
        print('class:',c)
        array = Array[:,:,:,c]
        shape = array.shape
        new_shape = np.array([int(np.round(817*0.25/0.625)),512,512])   # new_spacing (0.625, 0.625, 0.625)
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
        unique_labels = np.unique(array)
        reshaped = np.zeros(new_shape, dtype=dtype_data)

        for i, cl in enumerate(unique_labels):
            print(i,cl)
            reshaped_multihot = np.round(
                map_coordinates((array == cl).astype(float), coord_map, order=1, mode='nearest'))   # linear interpolation
            reshaped[reshaped_multihot > 0.5] = cl
        reshaped_final_data.append(reshaped[None].astype(dtype_data))
    
    # inverse one hot
    reshaped_final_data = np.array(reshaped_final_data, dtype=dtype_data)
    reshaped_final_data = np.squeeze(reshaped_final_data, axis=1)
    print(reshaped_final_data.shape, reshaped_final_data.dtype) # (c, N, 512,512) uint16
    Array = np.argmax(reshaped_final_data, axis=0).astype('uint16')
    print(Array.shape, Array.dtype)

    # save
    resampled_data = sitk.GetImageFromArray(Array)
    resampled_data.SetSpacing((0.625, 0.625, 0.625))
    sitk.WriteImage(resampled_data, dirpath)



def resample_data(datapath, dirpath):
    '''
        3d resampling for data, make that the voxel spacing of x,y,z axis is the same.

        :params datapath: the path of the data file    e.g. 'data/patient46.nii.gz'
        :parmas dirpath : the stored path of resampled data file 
    '''


    data = sitk.ReadImage(datapath)
    array = sitk.GetArrayFromImage(data)
    # print(array.shape)  # (817,512,512)
    # print(data.GetSize())   # (512,512,817)
    # print(data.GetSpacing()) # (0.625,0.625,0.25)
    dtype_data = array.dtype
    print(array.shape, array.dtype)             # int16
    print(np.unique(array))

    reshaped_final_data = []
    shape = array.shape
    new_shape = np.array([int(np.round(817*0.25/0.625)),512,512])
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
    reshaped_final_data = (map_coordinates(array, coord_map, order=3, mode='nearest').astype(dtype_data))    # done with third order spline
    print(reshaped_final_data.shape, reshaped_final_data.dtype) # ( N, 512,512) int16
    print(np.unique(reshaped_final_data))

    # save
    resampled_data = sitk.GetImageFromArray(reshaped_final_data)
    resampled_data.SetSpacing((0.625, 0.625, 0.625))
    sitk.WriteImage(resampled_data, dirpath)






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