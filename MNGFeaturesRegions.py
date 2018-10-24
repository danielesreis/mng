import math
import numpy as np

class MNGFeaturesRegions():

	def __init__(self, feature_means, n=5):
		self.feature_means 	= feature_means
		self.n 				= n

	def regions_means(self, img):

		def first_nonwhite_pixel(img, known_point, axis):
			data 	= img[known_point,:,0] if axis == 'y' else img[:,known_point,0]
			i 		= np.where(data != 255)[0][0]
			return i

		def last_nonwhite_pixel(img, known_point, axis):
			data 	= img[known_point,:,0] if axis == 'y' else img[:,known_point,0]
			i 		= np.where(data != 255)[0][-1]
			return i

		height, width, __ 	= img.shape
		slice_height 		= math.round(height/self.n)

		for i in range(self.n):
			skip 	= 1 if i > 0 else 0

			y_i 	= i * slice_height + skip
			y_f 	= y_i + slice_height

			x_0		= first_nonwhite_pixel(img, y_i, 'y')
			x_1		= first_nonwhite_pixel(img, y_f, 'y')
			x_i		= x_0 if x_0 > x_i else x_0

			x_0		= last_nonwhite_pixel(img, y_i, 'y')
			x_1		= last_nonwhite_pixel(img, y_f, 'y')
			x_f		= x_0 if x_0 < x_i else x_0

			means = self.feature_means.channels_mean(img[y_i:y_f,x_i:x_f,:])

			if i == 0:
				reg_means = np.array([[mean[0], mean[1], mean[2]]])
			else:
				reg_means = np.append(reg_means, [[mean[0], mean[1], mean[2]]], axis=0)

		return reg_means

	def regions_mean_diffs(self, img):	
		means = self.regions_means(img, self.n)

		for i in range(self.n-1):
			for j in np.arange(i,self.n-1):
				if i == 0 and j == 0:
					regions_diffs = np.array([means[i]-means[j+1]]) 
				else:
					regions_diffs = np.append(regions_diffs, [means[i]-means[j+1]], axis=0)

		return region_diffs

	def mean_diffs(self, img):

		if self.n == 1:
			means = self.feature_means.channels_mean(img)
			diffs = np.array([mean[0]-mean[1], mean[0]-mean[2], mean[1]-mean[2]])

		else:
			diffs = self.regions_mean_diffs(img, self.n)

		return diffs

	def apex_means(self, img, OFFSET_Y_apex_stalk=0.15, HEIGHT_FRACTION=0.05):

		def first_nonwhite_pixel(img, known_point, axis):
			data 	= img[known_point,:,0] if axis == 'y' else img[:,known_point,0]
			i 		= np.where(data != 255)[0][0]
			return i

		def last_nonwhite_pixel(img, known_point, axis):
			data 	= img[known_point,:,0] if axis == 'y' else img[:,known_point,0]
			i 		= np.where(data != 255)[0][-1]
			return i

		height, width, __ 		= img.shape

		y_c						= round(OFFSET_Y_apex_stalk * height)
		slice_height 			= round(height * HEIGHT_FRACTION / 2)
		y_f						= y_c + slice_height
		y_i						= y_c - slice_height

		x_0						= first_nonwhite_pixel(img, y_i, 'y')
		x_1						= first_nonwhite_pixel(img, y_f, 'y')
		x_i						= x_0 if x_0 > x_i else x_0

		x_0						= last_nonwhite_pixel(img, y_i, 'y')
		x_1						= last_nonwhite_pixel(img, y_f, 'y')
		x_f						= x_0 if x_0 < x_i else x_0

		means 		= self.feature_means.channels_mean(img[y_i:y_f, x_i:x_f])
		mean_apex 	= np.array([mean[0], mean[1], mean[2]])

		return mean_apex

	def equator_means(self, img, HEIGHT_FRACTION=0.05):

		def first_nonwhite_pixel(img, known_point, axis):
			data 	= img[known_point,:,0] if axis == 'y' else img[:,known_point,0]
			i 		= np.where(data != 255)[0][0]
			return i

		def last_nonwhite_pixel(img, known_point, axis):
			data 	= img[known_point,:,0] if axis == 'y' else img[:,known_point,0]
			i 		= np.where(data != 255)[0][-1]
			return i

		height, width, __ 		= img.shape

		y_c						= round(height / 2)
		slice_height 			= round(height * HEIGHT_FRACTION / 2)
		y_f						= y_c + slice_height
		y_i						= y_c - slice_height

		x_0						= first_nonwhite_pixel(img, y_i, 'y')
		x_1						= first_nonwhite_pixel(img, y_f, 'y')
		x_i						= x_0 if x_0 > x_i else x_0

		x_0						= last_nonwhite_pixel(img, y_i, 'y')
		x_1						= last_nonwhite_pixel(img, y_f, 'y')
		x_f						= x_0 if x_0 < x_i else x_0

		means 			= self.feature_means.channels_mean(img[y_i:y_f, x_i:x_f])
		mean_equator	= np.array([mean[0], mean[1], mean[2]])

		return mean_equator

	def stalk_means(self, img, OFFSET_Y_apex_stalk=0.15, HEIGHT_FRACTION=0.05):

		def first_nonwhite_pixel(img, known_point, axis):
			data 	= img[known_point,:,0] if axis == 'y' else img[:,known_point,0]
			i 		= np.where(data != 255)[0][0]
			return i

		def last_nonwhite_pixel(img, known_point, axis):
			data 	= img[known_point,:,0] if axis == 'y' else img[:,known_point,0]
			i 		= np.where(data != 255)[0][-1]
			return i

		height, width, __ 		= img.shape

		y_c						= height - round(OFFSET_Y_apex_stalk * height)
		slice_height 			= round(height * HEIGHT_FRACTION / 2)
		y_f						= y_c + slice_height
		y_i						= y_c - slice_height
		
		x_0						= first_nonwhite_pixel(img, y_i, 'y')
		x_1						= first_nonwhite_pixel(img, y_f, 'y')
		x_i						= x_0 if x_0 > x_i else x_0

		x_0						= last_nonwhite_pixel(img, y_i, 'y')
		x_1						= last_nonwhite_pixel(img, y_f, 'y')
		x_f						= x_0 if x_0 < x_i else x_0

		means 		= self.feature_means.channels_mean(img[y_i:y_f, x_i:x_f])
		mean_stalk 	= np.array([mean[0], mean[1], mean[2]])

		return mean_stalk

	def apex_equator_stalk_means(img):
		apex_m 		= self.apex_means(img)
		equator_m 	= self.equator_means(img)
		stalk_k 	= self.stalk_means(img)

		means = np.array([apex_m, equator_m, stalk_m])
		return means

	def regions_means_diffs(self, img):	
		means = apex_equator_stalk_means(img)		

		region_diffs = np.array([	means[0][0] - means[1][0], means[1][0] - means[2][0], means[0][0] - means[2][0],\
									means[0][1] - means[1][1], means[1][1] - means[2][1], means[0][1] - means[2][1],\
									means[0][2] - means[1][2], means[1][2] - means[2][2], means[0][2] - means[2][2]])

		return region_diffs

	def get_var_mean_names():
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

	def get_var_mean_diffs_names():
		channels_RGB = ['R', 'G', 'B']
		channels_HSV = ['H', 'S', 'V']
		channels_Lab = ['L', 'a', 'b']
		names_RGB = []
		names_HSV = []
		names_Lab = []

		for i in range(n-1):
			for j in range(i, n-1):
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

