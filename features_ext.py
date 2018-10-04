import pandas as pd
import features as ft
from segmentation import get_inverse_mask
import cv2
import os

path 			= os.getcwd() + '\\segmentation\\'
folders			= os.listdir(path)
dest_folder		= path + '\\..\\feature_extraction\\'

feature_names	= [ 'mean_R', 'mean_G', 'mean_B', 'mean_H', 'mean_S', 'mean_V', 'mean_L', 'mean_a', 'mean_b', 'RG_rate', 'RB_rate', \
					'SH_rate', 'dominant_HSV', 'RG_diff', 'RB_diff', 'GB_diff', 'apex_mean_R', 'apex_mean_G', 'apex_mean_B', \
					'equator_mean_R', 'equator_mean_G', 'equator_mean_B', 'stalk_mean_R', 'stalk_mean_G', 'stalk_mean_B', \
					'long_gradient_R', 'long_gradient_G', 'long_gradient_B', 'apex_equator_R_diff', 'equator_stalk_R_diff', \
					'apex_stalk_R_diff', 'apex_equator_G_diff', 'equator_stalk_G_diff', 'apex_stalk_G_diff', 'apex_equator_B_diff', \
					'equator_stalk_B_diff', 'apex_stalk_B_diff', 'bcd', 'cd', 'dd', 'area', 'diameter']

for seg_folder in folders:
	img_names 	= os.listdir(path + seg_folder)
	data 		= pd.DataFrame(columns = feature_names)

	for img_name in img_names:
		BGR_img 	= cv2.imread(img_name)
		gray_img 	= cv2.cvtColor(BGR_img, cv2.COLOR_BGR2GRAY)
		binary_img 	= get_inverse_mask(gray_img)
		
		RGB_img 	= BGR_img[:,:,::-1]
		HSV_img 	= cv2.cvtColor(RGB_img, cv2.COLOR_RGB2HSV)
		#check this
		Lab_img 	= cv2.cvtColor(RGB_img, cv2.COLOR_RGB2Lab)


		mean_R, mean_G, mean_B 								= ft.channels_mean(RGB_img)
		mean_H, mean_S, mean_B 								= ft.channels_mean(HSV_img)
		mean_L, mean_a, mean_b 								= ft.channels_mean(Lab_img)
		RG_rate, RB_rate, __ 								= ft.space_rates(RGB_img)
		# it's H/S actually
		SH_rate, __, __ 									= ft.space_rates(HSV_img)
		dominant_HSV										= ft.dominant_HSV_color(HSV_img)
		RG_diff, RB_diff, GB_diff							= ft.space_diffs(RGB_img)
		apex_mean_R, apex_mean_G, apex_mean_B 				= ft.apex_means(RGB_img)
		equator_mean_R, equator_mean_G, equator_mean_B 		= ft.equator_means(RGB_img)
		stalk_mean_R, stalk_mean_G, stalk_mean_B 			= ft.stalk_means(RGB_img)
		long_gradient_R, long_gradient_G, long_gradient_B 	= ft.longitudinal_gradient(RGB_img)
		regions_diffs										= ft.regions_mean_diffs(RGB_img)
		bcd  												= ft.box_counting_dimension(RGB_img)
		cd  												= ft.correlation_dimension(RGB_img)
		dd  												= ft.dilation_dimension(RGB_img)
		# area is the same as number of pixels
		area												= ft.estimated_area(binary_img)
		diameter											= ft.estimated_diameter(binary_img)

		feature_values	= [ mean_R, mean_G, mean_B, mean_H, mean_S, mean_V, mean_L, mean_a, mean_b, RG_rate, RB_rate, \
							SH_rate, dominant_HSV, RG_diff, RB_diff, GB_diff, apex_mean_R, apex_mean_G, apex_mean_B, \
							equator_mean_R, equator_mean_G, equator_mean_B, stalk_mean_R, stalk_mean_G, stalk_mean_B, \
							long_gradient_R, long_gradient_G, long_gradient_B, regions_diffs[0], regions_diffs[1], \
							regions_diffs[2], regions_diffs[3], regions_diffs[4], regions_diffs[5], regions_diffs[6], \
							regions_diffs[7], regions_diffs[8], bcd, cd, dd, n_pixels, area, diameter]

		feature_row = pd.Series(data = feature_values, index = feature_names, name = img_name)

		data = data.append(feature_row)

	data.to_csv(dest_folder + seg_folder + '.csv', sep = ';')

