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
					'apex_mean_H', 'apex_mean_S', 'apex_mean_V', 'equator_mean_H', 'equator_mean_S', 'equator_mean_V', 'stalk_mean_H', \
					'stalk_mean_S', 'stalk_mean_V', 'apex_mean_L', 'apex_mean_a', 'apex_mean_b', 'equator_mean_L', 'equator_mean_a', \
					'equator_mean_b', 'stalk_mean_L', 'stalk_mean_a', 'stalk_mean_b', 'long_gradient_R', 'long_gradient_G', \
					'long_gradient_B', 'apex_equator_R_diff', 'equator_stalk_R_diff', 'apex_stalk_R_diff', 'apex_equator_G_diff', \
					'equator_stalk_G_diff', 'apex_stalk_G_diff', 'apex_equator_B_diff', 'equator_stalk_B_diff', 'apex_stalk_B_diff', \
					'apex_equator_H_diff', 'equator_stalk_H_diff', 'apex_stalk_H_diff', 'apex_equator_S_diff', 'equator_stalk_S_diff', \
					'apex_stalk_S_diff', 'apex_equator_V_diff', 'equator_stalk_V_diff', 'apex_stalk_V_diff', 'apex_equator_L_diff', \
					'equator_stalk_L_diff', 'apex_stalk_L_diff', 'apex_equator_a_diff', 'equator_stalk_a_diff', 'apex_stalk_a_diff', \
					'apex_equator_b_diff', 'equator_stalk_b_diff', 'apex_stalk_b_diff', 'bcd', 'cd', 'dd', 'area', 'diameter']

for seg_folder in folders:
	img_names 	= os.listdir(path + seg_folder)
	data 		= pd.DataFrame(columns = feature_names)

	for img_name in img_names:
		BGR_img 	= cv2.imread(img_name)
		gray_img 	= cv2.cvtColor(BGR_img, cv2.COLOR_BGR2GRAY)
				
		RGB_img 	= BGR_img[:,:,::-1]
		HSV_img 	= cv2.cvtColor(RGB_img, cv2.COLOR_RGB2HSV)
		#check this
		Lab_img 	= cv2.cvtColor(RGB_img, cv2.COLOR_RGB2Lab)


		mean_R, mean_G, mean_B 								= ft.channels_mean(RGB_img)
		mean_H, mean_S, mean_B 								= ft.channels_mean(HSV_img)
		mean_L, mean_a, mean_b 								= ft.channels_mean(Lab_img)

		RG_rate, RB_rate, __ 								= ft.space_rates(RGB_img)
		SH_rate, __, __ 									= ft.space_rates(HSV_img)

		dominant_HSV										= ft.dominant_HSV_color(HSV_img)

		RG_diff, RB_diff, GB_diff							= ft.space_diffs(RGB_img)

		apex_mean_R, apex_mean_G, apex_mean_B 				= ft.apex_means(RGB_img)
		equator_mean_R, equator_mean_G, equator_mean_B 		= ft.equator_means(RGB_img)
		stalk_mean_R, stalk_mean_G, stalk_mean_B 			= ft.stalk_means(RGB_img)
		apex_mean_H, apex_mean_S, apex_mean_V 				= ft.apex_means(HSV_img)
		equator_mean_H, equator_mean_S, equator_mean_V 		= ft.equator_means(HSV_img)
		stalk_mean_H, stalk_mean_S, stalk_mean_V 			= ft.stalk_means(HSV_img)
		apex_mean_L, apex_mean_a, apex_mean_b 				= ft.apex_means(Lab_img)
		equator_mean_L, equator_mean_a, equator_mean_b 		= ft.equator_means(Lab_img)
		stalk_mean_L, stalk_mean_a, stalk_mean_b 			= ft.stalk_means(Lab_img)
		
		long_gradient_R, long_gradient_G, long_gradient_B 	= ft.longitudinal_gradient(RGB_img)

		regions_diffs_RGB									= ft.regions_mean_diffs(RGB_img)
		regions_diffs_HSV									= ft.regions_mean_diffs(HSV_img)
		regions_diffs_Lab									= ft.regions_mean_diffs(Lab_img)

		bcd  												= ft.box_counting_dimension(RGB_img)
		cd  												= ft.correlation_dimension(RGB_img)
		dd  												= ft.dilation_dimension(RGB_img)

		area												= ft.estimated_area(gray_img)
		diameter											= ft.estimated_diameter(gray_img)

		feature_values	= [ mean_R, mean_G, mean_B, mean_H, mean_S, mean_V, mean_L, mean_a, mean_b, RG_rate, RB_rate, \
							SH_rate, dominant_HSV, RG_diff, RB_diff, GB_diff, apex_mean_R, apex_mean_G, apex_mean_B, \
							equator_mean_R, equator_mean_G, equator_mean_B, stalk_mean_R, stalk_mean_G, stalk_mean_B, \
							apex_mean_H, apex_mean_S, apex_mean_V, equator_mean_H, equator_mean_S, equator_mean_V, stalk_mean_H,\
							stalk_mean_S, stalk_mean_V, apex_mean_L, apex_mean_a, apex_mean_b, equator_mean_L, equator_mean_a, \
							equator_mean_b, stalk_mean_L, stalk_mean_a, stalk_mean_b, long_gradient_R, long_gradient_G, long_gradient_B, \
							regions_diffs_RGB[0], regions_diffs_RGB[1], regions_diffs_RGB[2], regions_diffs_RGB[3], \
							regions_diffs_RGB[4], regions_diffs_RGB[5], regions_diffs_RGB[6], regions_diffs_RGB[7], \
							regions_diffs_RGB[8], regions_diffs_HSV[0], regions_diffs_HSV[1], regions_diffs_HSV[2], \
							regions_diffs_HSV[3], regions_diffs_HSV[4], regions_diffs_HSV[5], regions_diffs_HSV[6], \
							regions_diffs_HSV[7], regions_diffs_HSV[8], regions_diffs_Lab[0], regions_diffs_Lab[1], \
							regions_diffs_Lab[2], regions_diffs_Lab[3], regions_diffs_Lab[4], regions_diffs_Lab[5], \
							regions_diffs_Lab[6], regions_diffs_Lab[7], regions_diffs_Lab[8], \
							bcd, cd, dd, n_pixels, area, diameter]

		feature_row = pd.Series(data = feature_values, index = feature_names, name = img_name)

		data = data.append(feature_row)

	data.to_csv(dest_folder + seg_folder + '.csv', sep = ';')

