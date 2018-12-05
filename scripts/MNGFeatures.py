import cv2
import math
import numpy as np
import pandas as pd
from MNGFeaturesSize import MNGFeaturesSize
from MNGFeaturesMeans import MNGFeaturesMeans
from MNGFeaturesDominantColor import MNGFeaturesDominantColor
from MNGFeaturesRates import MNGFeaturesRates
from MNGFeaturesGradient import MNGFeaturesGradient
from MNGFeaturesFractal import MNGFeaturesFractal
from MNGFeaturesRegions import MNGFeaturesRegions

class MNGFeatures():

	feature_names	= [ 'mean_R_full', 'mean_G_full', 'mean_B_full',															\
						'mean_H_full', 'mean_S_full', 'mean_V_full',															\
						'mean_L_full', 'mean_a_full', 'mean_b_full',															\
						'area', 'height', 'width', 'diameter',																						\
						'dominant_HSV',																							\
						'RG_rate', 'RB_rate', 'SH_rate',																		\
						'long_gradient',																						\
						'bcd', 'cd', 'dd',																						\
						'RG_diff_full', 'RB_diff_full', 'GB_diff_full',																		\
						'apex_R', 'apex_G', 'apex_B', 'equator_R', 'equator_G', 'equator_B', 'stalk_R', 'stalk_G', 'stalk_B',	\
						'apex_equator_R_diff', 'equator_stalk_R_diff', 'apex_stalk_R_diff', 									\
						'apex_equator_G_diff', 'equator_stalk_G_diff', 'apex_stalk_G_diff',										\
						'apex_equator_B_diff', 'equator_stalk_B_diff', 'apex_stalk_B_diff']										\

	def new_df(self):
		self.data = pd.DataFrame(index=self.image_names, columns=[self.current_features], dtype=np.float64)

	def insert_feature_row(self, img_name, feature_values):
		feature_row = pd.Series(data=feature_values, index=[self.current_features], name=img_name.split('.')[0])
		self.data.loc[img_name.split('.')[0]] = feature_row

	def save_data(self):
		file_path = self.dest_folder + self.current_features_name + '_all.csv'
		self.data.to_csv(file_path, sep=';')
		file_path = self.edit_data_frame()
		return file_path

	def edit_feature_names(self):
		names_means = self.features_regions.get_var_mean_names()
		names_diffs = self.features_regions.get_var_mean_diffs_names()

		names = [names_means, names_diffs]
		names = [item for sublist in names for item in sublist]

		self.feature_names = [self.feature_names, names]
		self.feature_names = [item for sublist in self.feature_names for item in sublist]

	def edit_data_frame(self):
		new_data = self.data.copy()
		columns = self.data.columns
		index = self.data.index.values

		info = list()
		info = [ind.split('_') for ind in index]

		var = [inf[0][4:] for inf in info]
		sem = [int(inf[1][3:]) for inf in info]
		num = [int(inf[2].split('lado')[0][3:]) for inf in info]

		var = pd.Series(var, index)
		sem = pd.Series(sem, index)
		num = pd.Series(num, index)

		new_data['var'] = var
		new_data['sem'] = sem
		new_data['num'] = num

		means = new_data.groupby(['var', 'sem', 'num'])[columns].mean()
		new_data = means.reset_index().drop(columns=['var', 'sem', 'num'])

		new_index = [ind.split('lado')[0] for ind in index]
		new_index = new_index[::2]

		new_data['ind'] = new_index
		new_data.set_index('ind', inplace=True)

		file_path = self.dest_folder + self.current_features_name + '_all_half.csv'
		new_data.to_csv(file_path, sep=';')

		return file_path

	def add_target(self, file_path, target1):
		att1 = pd.read_csv(target1, sep=';')[self.att].values
		# att2 = pd.read_csv(target2, sep=';')[self.att].values

		# att = np.concatenate((att1,att2))
		data = pd.read_csv(file_path, sep=';', index_col=0)

		data.insert(loc=0, column=self.att, value=att1)
		data.to_csv(file_path, sep=';')

	@property
	def current_features(self):
		return self._current_features

	@current_features.setter
	def current_features(self, current_features):
		self._current_features = current_features

	@property
	def current_features_name(self):
		return self._current_features_name

	@current_features_name.setter
	def current_features_name(self, current_features_name):
		self._current_features_name = current_features_name
	
	# def get_feature_method(features):
	# 	indexes = list()

	# 	for index,name in enumerate(self.feature_names):
	# 		if name in self.current_features:
	# 			indexes.append(index)

	# 	methods = [self.feature_methods[index] for index in indexes]
	# 	return methods

	def __init__(self, folder, image_names, att, n=5):
		self.dest_folder 		= folder + '..\\features\\'
		self.image_names		= [image_name.split('.')[0] for image_name in image_names]
		self.att 				= att
		self.n 					= n

		self.features_means		= MNGFeaturesMeans()
		self.features_size 		= MNGFeaturesSize()
		self.features_dominant	= MNGFeaturesDominantColor(self.features_means)
		self.features_rates		= MNGFeaturesRates(self.features_means)
		self.features_gradient	= MNGFeaturesGradient()
		self.features_regions	= MNGFeaturesRegions(self.features_means, n)
		self.features_fractal	= MNGFeaturesFractal()

		self._current_features  	= None
		self._current_features_name = None
		self.data 					= None
		self.edit_feature_names()

	def extract_features(self, BGR_img, img_name):
		gray_img = cv2.cvtColor(BGR_img, cv2.COLOR_BGR2GRAY)
		
		RGB_img = BGR_img[:,:,::-1]
		HSV_img = cv2.cvtColor(RGB_img, cv2.COLOR_RGB2HSV)
		Lab_img = cv2.cvtColor(RGB_img, cv2.COLOR_RGB2Lab)

		if self.current_features == self.feature_names[:9]:
			means_RGB 				= self.features_means.channels_mean(RGB_img)
			means_HSV 				= self.features_means.channels_mean(HSV_img)
			means_Lab 				= self.features_means.channels_mean(Lab_img)

			feature_values = list(np.concatenate((means_RGB, means_HSV, means_Lab), axis=None))

		elif self.current_features == self.feature_names[9:13]:
			area					= self.features_size.estimated_area(gray_img)
			height, width			= self.features_size.estimated_measures(gray_img)
			diameter				= self.features_size.estimated_diameter(gray_img)

			feature_values = [area, height, width, diameter]

		elif self.current_features == self.feature_names[13]:
			dominant_HSV 			= self.features_dominant.dominant_HSV_color(HSV_img)
			feature_values = dominant_HSV

		elif self.current_features == self.feature_names[14:17]:
			rates_RGB		 		= self.features_rates.space_rates(RGB_img)
			rates_HSV				= self.features_rates.space_rates(HSV_img)

			feature_values = [rates_RGB[0], rates_RGB[1], rates_HSV[0]]

		elif self.current_features == self.feature_names[17]:
			long_gradient			= self.features_gradient.longitudinal_gradient(RGB_img)

			feature_values = long_gradient

		elif self.current_features == self.feature_names[18:21]:
			bcd		  	 			= self.features_fractal.box_counting_dimension(gray_img)
			cd		  				= self.features_fractal.correlation_dimension(gray_img)
			dd		  				= self.features_fractal.dilation_dimension(gray_img)

			feature_values = [bcd, cd, dd]

		elif self.current_features == self.feature_names[21:42]:
			means_diffs_full 					= self.features_regions.mean_diffs(RGB_img, 1)
			means_apex_equator_stalk 			= self.features_regions.apex_equator_stalk_means(RGB_img)
			mean_diffs_apex_equator_stalk_RGB	= self.features_regions.regions_means_diffs(RGB_img)

			feature_values = list(np.concatenate((means_diffs_full.flatten(), means_apex_equator_stalk.flatten(), mean_diffs_apex_equator_stalk_RGB.flatten()), axis=None))

		elif self.current_features == self.feature_names[42:]:
			means_n_RGB				= self.features_regions.regions_means(RGB_img, self.n)
			means_n_HSV				= self.features_regions.regions_means(HSV_img, self.n)
			means_n_Lab				= self.features_regions.regions_means(Lab_img, self.n)
			means_diffs_n_RGB		= self.features_regions.mean_diffs(RGB_img, self.n)
			means_diffs_n_HSV		= self.features_regions.mean_diffs(HSV_img, self.n)
			means_diffs_n_Lab		= self.features_regions.mean_diffs(Lab_img, self.n)

			feature_values = list(np.concatenate((means_n_RGB.flatten(), means_n_HSV.flatten(), means_n_Lab.flatten(), means_diffs_n_RGB.flatten(), means_diffs_n_HSV.flatten(), means_diffs_n_Lab.flatten()), axis=None))
			
		# elif self.current_features == self.feature_names:
		# 	means_RGB 				= self.features_means.channels_mean(RGB_img)
		# 	means_HSV 				= self.features_means.channels_mean(HSV_img)
		# 	means_Lab 				= self.features_means.channels_mean(Lab_img)

		# 	area					= self.features_sizes.estimated_area(gray_img)
		# 	diameter				= self.features_sizes.estimated_diameter(gray_img)

		# 	dominant_HSV		  	= self.features_dominant.dominant_HSV_color(HSV_img)

		# 	rates_RGB		 		= self.features_rates.space_rates(RGB_img)
		# 	rates_HSV				= self.features_rates.space_rates(HSV_img)

		# 	long_gradient			= self.features_gradient.longitudinal_gradient(RGB_img)

		# 	bcd		  	 			= self.features_fractal.box_counting_dimension(gray_img)
		# 	cd		  				= self.features_fractal.correlation_dimension(gray_img)
		# 	dd		  				= self.features_fractal.dilation_dimension(gray_img)

		# 	means_diffs_full 					= self.features_regions.mean_diffs(RGB_img, 1)
		# 	means_apex_equator_stalk 			= self.features_regions.apex_equator_stalk_means(img)
		# 	mean_diffs_apex_equator_stalk_RGB	= self.features_regions.regions_means_diffs(RGB_img)

		# 	means_n_RGB				= self.features_regions.regions_means(RGB_img, n)
		# 	means_n_HSV				= self.features_regions.regions_means(HSV_img, n)
		# 	means_n_Lab				= self.features_regions.regions_means(Lab_img, n)
		# 	means_diffs_n_RGB		= self.features_regions.mean_diffs(RGB_img, n)
		# 	means_diffs_n_HSV		= self.features_regions.mean_diffs(HSV_img, n)
		# 	means_diffs_n_Lab		= self.features_regions.mean_diffs(Lab_img, n)

		# 	feature_values = list(np.concatenate((	means_RGB, means_HSV, means_Lab, 																				\
		# 											[area], [diameter], [height], [width], 																			\
		# 											[dominant_HSV], 																								\
		# 											[rates_RGB[0]], [rates_RGB[1]], [rates_HSV[0]], 																\
		# 											[long_gradient], 																								\
		# 											[bcd], [cd], [dd], 																								\
		# 											means_diffs_full.flatten(), means_apex_equator_stalk.flatten(), mean_diffs_apex_equator_stalk_RGB.flatten(), 	\
		# 											means_n_RGB.flatten(), means_n_HSV.flatten(), means_n_Lab.flatten(),means_diffs_n_RGB.flatten(), 				\
		# 											means_diffs_n_HSV.flatten(), means_diffs_n_Lab.flatten()), axis=None))		

		self.insert_feature_row(img_name, feature_values)
