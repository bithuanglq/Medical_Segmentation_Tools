from skimage.transform import resize
from batchgenerators.augmentations.utils import resize_segmentation
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple



def downsampling_3d(datapath:str, dirpath:str, target_size:Tuple[int, int, int], is_label=False):
    '''
        datapath: e.g.  1.nii.gz
        dirpath: e.g.   11.nii.gz
        target_size: (x, y, z) e.g. (128, 128, 96)
        this fuc is only useful for 2d sampling, with the z axis unchanged
    '''

    data_image = sitk.ReadImage(datapath)
    data = sitk.GetArrayFromImage(data_image)   # (96,576,576) (z, y, x)
    data = np.transpose(data, (2, 1, 0))    # (x, y, z)
    size = data_image.GetSize()             # (x, y, z)
    spacing = data_image.GetSpacing()
    new_spacing = (spacing[0]*size[0]/target_size[0], spacing[1]*size[1]/target_size[1], spacing[2]*size[2]/target_size[2])
    dtype_data = data.dtype

    if is_label:
        new_data = []
        resize_fn = resize_segmentation
        order = 0
        new_shape_2d = [target_size[0], target_size[1]]
        axis = 2
        for slice_id in range(data.shape[axis]):
            new_data.append(resize_fn(data[slice_id], new_shape_2d, order).astype(dtype_data))
        new_data = np.stack(new_data, axis)
    else:
        new_data = []
        resize_fn = resize
        order = 3
        new_shape_2d = [target_size[0], target_size[1]]
        axis = 2
        for slice_id in range(data.shape[axis]):
            new_data.append(resize_fn(data[slice_id], new_shape_2d, order).astype(dtype_data))
        new_data = np.stack(new_data, axis)

    # save
    new_data_image = sitk.GetImageFromArray(new_data)
    new_data_image.SetSpacing(new_spacing)
    sitk.WriteImage(new_data_image, dirpath)


def upsampling_3d(datapath:str, dirpath:str, target_size:Tuple[int, int, int], is_label=False):
    '''
        datapath: e.g.  1.nii.gz
        dirpath: e.g.   11.nii.gz
        target_size: (x, y, z)
        this func can be used for 3d sampling, with z axis changed
    '''
    data_image = sitk.ReadImage(datapath)
    data = sitk.GetArrayFromImage(data_image)   # (96,128,128) (z, y, x)
    data = np.transpose(data, (2, 1, 0))    # (x, y, z)
    size = data_image.GetSize()             # (x, y, z)
    spacing = data_image.GetSpacing()
    new_spacing = (spacing[0]*size[0]/target_size[0], spacing[1]*size[1]/target_size[1], spacing[2]*size[2]/target_size[2])

    if is_label:
        new_data = torch.from_numpy(data.astype('float64')).unsqueeze(0).unsqueeze(0)       # (1, 1, x, y, z)
        new_data = F.interpolate(new_data, size=target_size, mode='nearest')
        new_data = torch.squeeze(new_data, dim=0).squeeze(0).numpy().astype('uint16')       # (x, y, z)
    else:
        new_data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
        new_data = F.interpolate(new_data, size=target_size, mode='trilinear')
        new_data = torch.squeeze(new_data, dim=0).squeeze(0).numpy()

    # save
    new_data_image = sitk.GetImageFromArray(new_data)
    new_data_image.SetSpacing(new_spacing)
    sitk.WriteImage(new_data_image, dirpath)
