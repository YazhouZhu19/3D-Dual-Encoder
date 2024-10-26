import numpy as np
import os

image_path = r"C:/Users/Windows/Desktop/ZYZ/BraTS/BraTS_training_data/preprocessed data/finalImage350.npy"
mask_path = r"C:/Users/Windows/Desktop/ZYZ/BraTS/BraTS_training_data/preprocessed data/finalMask350.npy"

image = np.load(image_path)
mask = np.load(mask_path)

print(np.shape(image))
print(np.shape(mask))



