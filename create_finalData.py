import os
import numpy as np


imgs_path = r"C:/Users/Windows/Desktop/ZYZ/BraTS/BraTS2020/data/preprocessed data/image/"
masks_path = r"C:/Users/Windows/Desktop/ZYZ/BraTS/BraTS2020/data/preprocessed data/mask/"
finalImg_savePath = r"C:/Users/Windows/Desktop/ZYZ/BraTS/BraTS2020/data/preprocessed data/"
finalMask_savePath = r"C:/Users/Windows/Desktop/ZYZ/BraTS/BraTS2020/data/preprocessed data/"

imgs_list = os.listdir(imgs_path)
masks_list = os.listdir(masks_path)

final_array = []
final_mask = []
for img in imgs_list:
    img_name = imgs_path + img
    img_array = np.load(img_name)
    final_array.append(img_array)
    print("the image:", img, "is appended")
finalImg_saveName = finalImg_savePath + "finalImage.npy"
np.save(finalImg_saveName, final_array)

for mask in masks_list:
    mask_name = masks_path + mask
    mask_array = np.load(mask_name)
    final_mask.append(mask_array)
    print("the image:", mask, "is appended")
finalMask_saveName = finalMask_savePath + "finalMask.npy"
np.save(finalMask_saveName, final_mask)





