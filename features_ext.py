import pandas as pd
import features as ft
import cv2
import os

path 			= os.getcwd() + '\\segmentation'
folders			= os.listdir(path)
dest_folder		= path + '\\..\\feature_extraction'
# incomplete
feature_names	= ['mean_R', 'mean_G', 'mean_B', 'mean_H', 'mean_S', 'mean_V', 'mean_L', 'mean_a', 'mean_b', 'dominant_HSV', 'n_pixels', 'area', 'diameter']

for seg_folder in folders:
	img_names 	= os.listdir(path + '\\' + folder)
	data 		= pd.DataFrame(columns = features)

	for img_name in img_names:
		img 		= cv2.imread(img_name)

		# here I call the the functions for each feature and append it to feature_values list
		# feature_values.append()
		feature_row = pd.Series(data = feature_values, index = feature_names, name = img_name)

		data = data.append(feature_row)

	data.to_csv(dest_folder + '\\' + seg_folder + '.csv', sep = ';')

