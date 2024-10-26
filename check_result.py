import numpy as np
import SimpleITK as itk
import os
import torch
import matplotlib.pyplot as plt


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


"""
raw_imagePath = r"C:/Users/Windows/Desktop/ZYZ/BraTS/BraTS_training_data/MICCAI_BraTS2020_TrainingData/BraTS20_Training_002/BraTS20_Training_002_t1ce.nii.gz"
image = itk.GetArrayFromImage(itk.ReadImage(raw_imagePath))
image = normalize(image)
print(np.max(image))
print(np.min(image))
"""


"""
image_path = r"C:/Users/Windows/Desktop/ZYZ/BraTS/BraTS_training_data/preprocessed data/image/BraTS20_Training_004.npy"
image = np.load(image_path)  # (160, 160, 160, 4)
image = np.transpose(image, (3, 0, 1, 2))
image = image[0][50]

plt.imshow(image, "gray")
plt.show()
"""




"""
raw_imagePath = r"C:/Users/Windows/Desktop/ZYZ/BraTS/BraTS_training_data/MICCAI_BraTS2020_TrainingData/BraTS20_Training_002/BraTS20_Training_002_flair.nii.gz"
image = itk.GetArrayFromImage(itk.ReadImage(raw_imagePath))
image = image[50]
plt.imshow(image, "gray")
plt.show()
"""


result_path = "./final_output_500.npy"
result = np.load(result_path)  # (1, 3, 160, 160, 160)

print(np.shape(result))


# result = np.reshape(result, (1,  3, 160, 160, 160))

result = result[0][2][70]

# result = result[0][2][50]
plt.imshow(result, "gray")
plt.show()










