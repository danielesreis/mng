import os
import numpy as np

current_path = os.getcwd()
folders = os.listdir(current_path)
nums = np.arange(1, 8)

for folder,num in zip(folders, nums):
	imgs = os.listdir(current_path + '\\' + folder)

	for img in imgs:
		img_path = current_path + '\\' + folder + '\\' + img
		img_new_path = current_path + '\\' + folder.split('_')[0] + str(num) + '_' + img.split('.')[0] + '_' + folder.split('_')[1] + '.jpg'
		os.rename(img_path, img_new_path)