import cv2
import math
import numpy as np
import pandas as pd
import MNGFeaturesSize
import MNGFeaturesMean
import MNGFeaturesDominantColor
import MNGFeaturesRates
import MNGFeaturesGradient
import MNGFeaturesFractal
import MNGFeaturesRegions

class MNGFeatures():

	feature_names	= [ 'mean_R_full', 'mean_G_full', 'mean_B_full',															\
						'mean_H_full', 'mean_S_full', 'mean_V_full',															\
						'mean_L_full', 'mean_a_full', 'mean_b_full',															\
						'area', 'diameter',																						\
						'dominant_HSV',																							\
						'RG_rate', 'RB_rate', 'SH_rate',																		\
						'long_gradient',																						\
						'bcd', 'cd', 'dd',																						\
						'RG_diff', 'RB_diff', 'GB_diff',																		\
						'apex_R', 'apex_G', 'apex_B', 'equator_R', 'equator_G', 'equator_B', 'stalk_R', 'stalk_G', 'stalk_B',	\
						'apex_equator_R_diff', 'equator_stalk_R_diff', 'apex_stalk_R_diff', 									\
						'apex_equator_G_diff', 'equator_stalk_G_diff', 'apex_stalk_G_diff',										\
						'apex_equator_B_diff', 'equator_stalk_B_diff', 'apex_stalk_B_diff']										\

	def __init__(self, folder, image_names):
		self.dest_folder 		= folder + '..\\features\\'
		self.image_names		= [image_name.split('.')[0] for image_name in image_names]
		self.data 				= new_df()

		self.features_means		= MNGFeaturesMeans()
		self.features_size 		= MNGFeaturesSize()
		self.features_dominant	= MNGFeaturesDominantColor(self.features_means)
		self.features_rates		= MNGFeaturesRates(self.features_means)
		self.features_gradient	= MNGFeaturesGradient()
		self.features_regions	= MNGFeaturesRegions(self.features_means, 5)
		self.features_fractal	= MNGFeaturesFractal()
		self.edit_feature_names()

	def new_df(self):
		self.data = pd.DataFrame(index=self.image_names, columns=self.feature_names)

	def insert_feature_row(self, img_name, feature_values):
		feature_row = pd.Series(data=feature_values, index=self.feature_names, name=img_name)
		self.data 	= self.data.append(feature_row)

	def save_data(self):
		file_path = self.dest_folder + 'features.csv'
		data.to_csv(file_path, sep=';')
		return file_path

	def extract_features(self, BGR_img, img_name):
		gray_img= cv2.cvtColor(BGR_img, cv2.COLOR_BGR2GRAY)
		
		RGB_img = BGR_img[:,:,::-1]
		HSV_img = cv2.cvtColor(RGB_img, cv2.COLOR_RGB2HSV)
		Lab_img = cv2.cvtColor(RGB_img, cv2.COLOR_RGB2Lab)

		means_RGB 				= self.features_means.channels_mean(RGB_img)
		means_HSV 				= self.features_means.channels_mean(HSV_img)
		means_Lab 				= self.features_means.channels_mean(Lab_img)

		area					= self.features_sizes.estimated_area(gray_img)
		diameter				= self.features_sizes.estimated_diameter(gray_img)
		
		dominant_HSV		  	= self.features_dominant.dominant_HSV_color(HSV_img)

		rates_RGB		 		= self.features_rates.space_rates(RGB_img)
		rates_HSV				= self.features_rates.space_rates(HSV_img)

		long_gradient			= self.features_gradient.longitudinal_gradient(RGB_img)

		bcd		  	 			= self.features_fractal.box_counting_dimension(gray_img)
		cd		  				= self.features_fractal.correlation_dimension(gray_img)
		dd		  				= self.features_fractal.dilation_dimension(gray_img)

		means_diffs_full 		= self.features_regions.mean_diffs(RGB_img, 1)

		means_apex_equator_stalk 			= self.features_regions.apex_equator_stalk_means(img)
		mean_diffs_apex_equator_stalk_RGB	= self.features_regions.regions_means_diffs(RGB_img)

		means_n_RGB				= self.features_regions.regions_means(RGB_img, n)
		means_n_HSV				= self.features_regions.regions_means(HSV_img, n)
		means_n_Lab				= self.features_regions.regions_means(Lab_img, n)

		means_diffs_n_RGB		= self.features_regions.mean_diffs(RGB_img, n)
		means_diffs_n_HSV		= self.features_regions.mean_diffs(HSV_img, n)
		means_diffs_n_Lab		= self.features_regions.mean_diffs(Lab_img, n)
		
		feature_values = [	mean_RGB.flatten(), means_HSV.flatten(), means_Lab.flatten(), 							\
							area, diameter, 																		\
							dominant_HSV, 																			\
							rates_RGB[0], rates_RGB[1], rates_HSV[0], 												\
							long_gradient, 																			\
							bcd, cd, dd, 																			\
							means_diffs_full.flatten(), 															\
							means_apex_equator_stalk.flatten(), 													\
							mean_diffs_apex_equator_stalk_RGB.flatten(), 											\
							means_n_RGB.flatten(), means_n_HSV.flatten(), means_n_Lab.flatten(), 					\
							means_diffs_n_RGB.flatten(), means_diffs_n_HSV.flatten(), means_diffs_n_Lab.flatten()	]

		self.insert_feature_row(img_name, feature_values)

	def edit_feature_names():
		names_means = self.features_regions.get_var_mean_names()
		names_diffs = self.features_regions.get_var_mean_diffs_names()

		names = [names_means, neames_diffs]
		names = [item for sublist in names for item in sublist]

		self.feature_names = [self.feature_names, names]
		self.feature_names = [item for sublist in self.feature_names for item in sublist]