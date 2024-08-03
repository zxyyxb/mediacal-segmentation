"""
此文件用于转换图像数据格式，将.nii.gz和.dcm文件转换为.npy文件
"""


# 基础库
import numpy as np
import os

# 图像.dcm文件读取
import pydicom
# 标签.nii.gz文件读取
import nibabel as nib

# .nii.gz文件转array
def nii2array(nii_path):
    img = nib.load(nii_path)
    img_array = np.asarray(img.get_fdata())
    img_array = np.transpose(img_array, (2, 1, 0))  # (x, y, z)
    img_array = np.around(img_array)
    img_array = img_array.astype(np.int16)
    return img_array

# .dcm文件转array
def dcm2array(dcm_path):
    slices = [pydicom.read_file(dcm_path + '/' + s) for s in os.listdir(dcm_path)]
    # 按照z轴的位置排序
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    # 转换为np.array
    image = np.stack([s.pixel_array for s in slices])  # (z, y, x)
    #将外部补全的部分修改成空气的数值
    image[image < 0] = 0
    return image.astype(np.int16)
