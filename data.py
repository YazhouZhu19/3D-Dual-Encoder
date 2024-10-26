# this is the code for pre-process
import SimpleITK as itk
import numpy as np
import os
# import nibabel as nib
from nipype.interfaces.ants import N4BiasFieldCorrection

image_path = r"C:/Users/Windows/Desktop/ZYZ/BraTS/BraTS2020/data/MICCAI_BraTS2020_TrainingData/"
image_save_path = r"C:/Users/Windows/Desktop/ZYZ/BraTS/BraTS2020/data/preprocessed data/image/"
mask_save_path = r"C:/Users/Windows/Desktop/ZYZ/BraTS/BraTS2020/data/preprocessed data/mask/"


def normalize(img, bottom=99, down=1):
    b = np.percentile(img, bottom)  # 获取上限
    t = np.percentile(img, down)  # 获取下限
    img = np.clip(img, t, b)  # 像素值裁剪
    img_nonzero = img[np.nonzero(img)]  # 只对黑色背景区域
    if np.std(img) == 0 or np.std(img_nonzero) == 0:
        return img
    else:
        tmp = (img - np.mean(img_nonzero)) / np.std(img_nonzero)
        tmp[tmp == np.min(tmp)] = -9
        return tmp


def crop(img, croph, cropw):
    height, width = img[0].shape
    starth = height//2-(croph//2)
    startw = width//2-(cropw//2)
    return img[:, starth:starth+croph, startw:startw+cropw]


def N4BiasFieldCorrect(filename, output_filename):
    normalized = N4BiasFieldCorrection()
    normalized.inputs.input_image = filename
    normalized.inputs.output_image = output_filename
    normalized.run()
    return None


fileList = os.listdir(image_path)
for filename in fileList:
    t1ce_name = image_path + filename + "/" + filename + "_t1ce.nii.gz"
    t1_name = image_path + filename + "/" + filename + "_t1.nii.gz"
    t2_name = image_path + filename + "/" + filename + "_t2.nii.gz"
    flair_name = image_path + filename + "/" + filename + "_flair.nii.gz"
    mask_name = image_path + filename + "/" + filename + "_seg.nii.gz"

    # 读取数据 the type of img is int16, mask is uint8 size is (155, 240, 240)
    t1ce_img = itk.GetArrayFromImage(itk.ReadImage(t1ce_name))
    t1_img = itk.GetArrayFromImage(itk.ReadImage(t1_name))
    t2_img = itk.GetArrayFromImage(itk.ReadImage(t2_name))
    flair_img = itk.GetArrayFromImage(itk.ReadImage(flair_name))
    mask_array = itk.GetArrayFromImage(itk.ReadImage(mask_name))

    # 加入切片 将图像和mask大小改变成(160, 240, 240)
    blackslices = np.zeros([240, 240])
    t1ce_img = np.insert(t1ce_img, 0, blackslices, axis=0)  # 在前面叠加
    t1ce_img = np.insert(t1ce_img, 0, blackslices, axis=0)
    t1ce_img = np.insert(t1ce_img, 0, blackslices, axis=0)
    t1ce_img = np.insert(t1ce_img, t1ce_img.shape[0], blackslices, axis=0)  # 在后面叠加
    t1ce_img = np.insert(t1ce_img, t1ce_img.shape[0], blackslices, axis=0)

    t1_img = np.insert(t1_img, 0, blackslices, axis=0)
    t1_img = np.insert(t1_img, 0, blackslices, axis=0)
    t1_img = np.insert(t1_img, 0, blackslices, axis=0)
    t1_img = np.insert(t1_img, t1_img.shape[0], blackslices, axis=0)
    t1_img = np.insert(t1_img, t1_img.shape[0], blackslices, axis=0)

    t2_img = np.insert(t2_img, 0, blackslices, axis=0)
    t2_img = np.insert(t2_img, 0, blackslices, axis=0)
    t2_img = np.insert(t2_img, 0, blackslices, axis=0)
    t2_img = np.insert(t2_img, t2_img.shape[0], blackslices, axis=0)
    t2_img = np.insert(t2_img, t2_img.shape[0], blackslices, axis=0)

    flair_img = np.insert(flair_img, 0, blackslices, axis=0)
    flair_img = np.insert(flair_img, 0, blackslices, axis=0)
    flair_img = np.insert(flair_img, 0, blackslices, axis=0)
    flair_img = np.insert(flair_img, flair_img.shape[0], blackslices, axis=0)
    flair_img = np.insert(flair_img, flair_img.shape[0], blackslices, axis=0)

    mask_array = np.insert(mask_array, 0, blackslices, axis=0)
    mask_array = np.insert(mask_array, 0, blackslices, axis=0)
    mask_array = np.insert(mask_array, 0, blackslices, axis=0)
    mask_array = np.insert(mask_array, mask_array.shape[0], blackslices, axis=0)
    mask_array = np.insert(mask_array, mask_array.shape[0], blackslices, axis=0)

    # 数据标准化
    t1ce_img = normalize(t1ce_img)
    t1_img = normalize(t1_img)
    t2_img = normalize(t2_img)
    flair_img = normalize(flair_img)
    print("the normalization is done")

    # crop
    t1ce_img = crop(t1ce_img, 160, 160)
    t1_img = crop(t1_img, 160, 160)
    t2_img = crop(t2_img, 160, 160)
    flair_img = crop(flair_img, 160, 160)
    mask_array = crop(mask_array, 160, 160)
    print("the crop is done")

    # 通道合并
    imagez, height, width = np.shape(flair_img)[0], np.shape(flair_img)[1], np.shape(flair_img)[2]
    fourModelArray = np.zeros((imagez, height, width, 4), np.float32)  # (160, 160, 160, 4)
    flair_img = flair_img.astype(np.float32)
    fourModelArray[:, :, :, 0] = flair_img
    t1_img = t1_img.astype(np.float32)
    fourModelArray[:, :, :, 1] = t1_img
    t1ce_img = t1ce_img.astype(np.float32)
    fourModelArray[:, :, :, 2] = t1ce_img
    t2_img = t2_img.astype(np.float32)
    fourModelArray[:, :, :, 3] = t2_img
    print("the final image array is done")
    imageName = image_save_path + filename + ".npy"
    np.save(imageName, fourModelArray)
    print("the image", filename, "is saved")


    # mask的处理
    """
    我们需要分割肿瘤区域有三个，ED(edema-label 2), ET(enhancing tumor-label 4), NET(tumor core-label 1)
    将label重新划分成WT(whole tumor), TC(tumor core), ET(enhancing tumor)
    WT (whole tumor)是整个肿瘤区域包含了ED, ET和NET三个部分
    TC(tumor core)是肿瘤核心区域,包含了ET和NET两个部分
    ET(enhancing tumor)是增强肿瘤区域，TC去除ET后剩下就是ET
    """
    wt_tc_etMaskArray = np.zeros((imagez, height, width, 3), np.uint8)
    WT_Label = mask_array.copy()  # whole tumor label
    WT_Label[mask_array == 1] = 1
    WT_Label[mask_array == 4] = 1
    WT_Label[mask_array == 2] = 1

    TC_Label = mask_array.copy()
    TC_Label[mask_array == 1] = 1
    TC_Label[mask_array == 2] = 0
    TC_Label[mask_array == 4] = 1

    ET_Label = mask_array.copy()
    ET_Label[mask_array == 1] = 0
    ET_Label[mask_array == 2] = 0
    ET_Label[mask_array == 4] = 1

    wt_tc_etMaskArray[:, :, :, 0] = WT_Label
    wt_tc_etMaskArray[:, :, :, 1] = TC_Label
    wt_tc_etMaskArray[:, :, :, 2] = ET_Label
    print("the overall mask is done")
    maskName = mask_save_path + filename + ".npy"
    np.save(maskName, wt_tc_etMaskArray)
    print("the mask for", filename, "is saved")
































"""
    t1ce_img = normalize(t1ce_img)
    t1ce_img = crop(t1ce_img, 160, 160)
    print(np.shape(t1ce_img))
"""












































"""
    # read images
    t1ceItkImage = itk.Cast(itk.ReadImage(t1ce_name), itk.sitkFloat32)
    t1ceArray = itk.GetArrayFromImage(t1ceItkImage)

    t1ItkImage = itk.Cast(itk.ReadImage(t1_name), itk.sitkFloat32)
    t1Array = itk.GetArrayFromImage(t1ItkImage)

    t2ItkImage = itk.Cast(itk.ReadImage(t2_name), itk.sitkFloat32)
    t2Array = itk.GetArrayFromImage(t2ItkImage)

    flairItkImage = itk.Cast(itk.ReadImage(flair_name), itk.sitkFloat32)
    flairArray = itk.GetArrayFromImage(flairItkImage)
"""




