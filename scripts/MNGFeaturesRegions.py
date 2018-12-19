import math
import cv2
import numpy as np

class MNGFeaturesRegions():

	def __init__(self, feature_means, n):
		self.feature_means 	= feature_means
		self.n 				= n

	def regions_means(self, img, gray_img, n):
		height, width, __ 	= img.shape
		slice_height 		= math.floor(height/n)

		y_i = 0
		y_f = slice_height

		means = self.feature_means.channels_mean(img[y_i:y_f,:,:])
		reg_means = np.array([[means[0], means[1], means[2]]])

		skip = 1
		for i in range(1,n-1,1):
			y_i 	= i * slice_height + skip
			y_f 	= y_i + slice_height

			means = self.feature_means.channels_mean(img[y_i:y_f,:,:])
			reg_means = np.append(reg_means, [[means[0], means[1], means[2]]], axis=0)

		y_i = y_f + 1

		means = self.feature_means.channels_mean(img[y_i:,:,:])
		reg_means = np.append(reg_means, [[means[0], means[1], means[2]]], axis=0)

		return reg_means

	def regions_mean_diffs(self, img, gray_img, n):	
		means = self.regions_means(img, gray_img, n)

		for i in range(n-1):
			for j in np.arange(i,n-1):
				if i == 0 and j == 0:
					regions_diffs = np.array([means[i]-means[j+1]]) 
				else:
					regions_diffs = np.append(regions_diffs, [means[i]-means[j+1]], axis=0)

		return regions_diffs

	def mean_diffs(self, img, gray_img, n):
		if n == 1:
			means = self.feature_means.channels_mean(img)
			diffs = np.array([means[0]-means[1], means[0]-means[2], means[1]-means[2]])

		else:
			diffs = self.regions_mean_diffs(img, gray_img, n)

		return diffs

	def apex_means(self, img, OFFSET_Y_apex_stalk=0.15, HEIGHT_FRACTION=0.05):

		def first_last_nonwhite_pixel(known_point):
			data = img[known_point,:]
			nonwhite_pixels = np.where(data != 255)[0]

			return nonwhite_pixels[0], nonwhite_pixels[-1]

		height, width, __ 	= img.shape

		y_c					= round(OFFSET_Y_apex_stalk * height)
		slice_height 		= round(height * HEIGHT_FRACTION / 2)
		y_f					= y_c + slice_height
		y_i					= y_c - slice_height

		xi_0,xi_1			= first_last_nonwhite_pixel(y_i)
		xf_0,xf_1			= first_last_nonwhite_pixel(y_f)
		
		x_i					= xi_0 if xi_0 > xf_0 else xf_0
		x_f					= xi_1 if xi_1 < xf_1 else xf_1

		means 		= self.feature_means.channels_mean(img[y_i:y_f,x_i:x_f,:])
		mean_apex 	= np.array([means[0], means[1], means[2]])

		return mean_apex

	def equator_means(self, img, HEIGHT_FRACTION=0.05):

		def first_last_nonwhite_pixel(known_point):
			data = img[known_point,:]
			nonwhite_pixels = np.where(data != 255)[0]

			return nonwhite_pixels[0], nonwhite_pixels[-1]

		height, width, __ 	= img.shape

		y_c					= round(height / 2)
		slice_height 		= round(height * HEIGHT_FRACTION / 2)
		y_f					= y_c + slice_height
		y_i					= y_c - slice_height

		xi_0,xi_1			= first_last_nonwhite_pixel(y_i)
		xf_0,xf_1			= first_last_nonwhite_pixel(y_f)

		x_i					= xi_0 if xi_0 > xf_0 else xf_0
		x_f					= xi_1 if xi_1 < xf_1 else xf_1

		means 			= self.feature_means.channels_mean(img[y_i:y_f,x_i:x_f,:])
		mean_equator	= np.array([means[0], means[1], means[2]])

		return mean_equator

	def stalk_means(self, img, OFFSET_Y_apex_stalk=0.15, HEIGHT_FRACTION=0.05):

		def first_last_nonwhite_pixel(known_point):
			data = img[known_point,:]
			nonwhite_pixels = np.where(data != 255)[0]

			return nonwhite_pixels[0], nonwhite_pixels[-1]

		height, width, __ 	= img.shape

		y_c					= height - round(OFFSET_Y_apex_stalk * height)
		slice_height 		= round(height * HEIGHT_FRACTION / 2)
		y_f					= y_c + slice_height
		y_i					= y_c - slice_height
		
		xi_0,xi_1			= first_last_nonwhite_pixel(y_i)
		xf_0,xf_1			= first_last_nonwhite_pixel(y_f)

		x_i					= xi_0 if xi_0 > xf_0 else xf_0
		x_f					= xi_1 if xi_1 < xf_1 else xf_1

		means 		= self.feature_means.channels_mean(img[y_i:y_f,x_i:x_f,:])
		mean_stalk 	= np.array([means[0], means[1], means[2]])

		return mean_stalk

	def apex_equator_stalk_means(self, img):
		apex_m 		= self.apex_means(img)
		equator_m 	= self.equator_means(img)
		stalk_m 	= self.stalk_means(img)

		means = np.array([apex_m, equator_m, stalk_m])
		return means

	def regions_means_diffs(self, img):	
		means = self.apex_equator_stalk_means(img)		

		region_diffs = np.array([	means[0][0] - means[1][0], means[1][0] - means[2][0], means[0][0] - means[2][0],\
									means[0][1] - means[1][1], means[1][1] - means[2][1], means[0][1] - means[2][1],\
									means[0][2] - means[1][2], means[1][2] - means[2][2], means[0][2] - means[2][2]])

		return region_diffs

	def get_var_mean_names(self):
		channels_RGB = ['R', 'G', 'B']
		channels_HSV = ['H', 'S', 'V']
		channels_Lab = ['L', 'a', 'b']
		names_RGB = []
		names_HSV = []
		names_Lab = []

		for region in range(self.n):
			for channel in channels_RGB:
				names_RGB.append('region_' + str(region) + '_' + channel)

			for channel in channels_HSV:
				names_HSV.append('region_' + str(region) + '_' + channel)

			for channel in channels_Lab:
				names_Lab.append('region_' + str(region) + '_' + channel)

		names = [names_RGB, names_HSV]
		names = [item for sublist in names for item in sublist]
		names = [names, names_Lab]
		names = [item for sublist in names for item in sublist]

		return names

	def get_var_mean_diffs_names(self):
		channels_RGB = ['R', 'G', 'B']
		channels_HSV = ['H', 'S', 'V']
		channels_Lab = ['L', 'a', 'b']
		names_RGB = []
		names_HSV = []
		names_Lab = []

		for i in range(self.n-1):
			for j in range(i, self.n-1):
				for channel in channels_RGB:
					names_RGB.append(str(i) + '_' + str(j+1) + '_' + channel + '_diff')

				for channel in channels_HSV:
					names_HSV.append(str(i) + '_' + str(j+1) + '_' + channel + '_diff')

				for channel in channels_Lab:
					names_Lab.append(str(i) + '_' + str(j+1) + '_' + channel + '_diff')

		names = [names_RGB, names_HSV]
		names = [item for sublist in names for item in sublist]
		names = [names, names_Lab]
		names = [item for sublist in names for item in sublist]

		return names

