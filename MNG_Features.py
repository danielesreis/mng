import cv2
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

class MNG_Features():

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

	def __init__(self, folder, image_names):
		self.dest_folder 	= folder + '..\\features\\'
		self.image_names	= image_names
		self.data 			= new_df()

	def new_df(self):
		self.data = pd.DataFrame(index=self.image_names, columns=self.feature_names)

	def insert_feature_row(self, img_name, feature_values):
		feature_row = pd.Series(data=feature_values, index=self.feature_names, name=img_name)
		self.data 	= self.data.append(feature_row)

	def save_data(self, subfolder):
		file_path = self.dest_folder + subfolder + '.csv'
		data.to_csv(file_path, sep=';')
		return file_path

	def extract_features(self, BGR_img, img_name):
		
		gray_img 	= cv2.cvtColor(BGR_img, cv2.COLOR_BGR2GRAY)
				
		RGB_img 		= BGR_img[:,:,::-1]
		HSV_img 		= cv2.cvtColor(RGB_img, cv2.COLOR_RGB2HSV)
		#check this
		Lab_img 		= cv2.cvtColor(RGB_img, cv2.COLOR_RGB2Lab)

		mean_R, mean_G, mean_B 								= self.channels_mean(RGB_img)
		mean_H, mean_S, mean_B 								= self.channels_mean(HSV_img)
		mean_L, mean_a, mean_b 								= self.channels_mean(Lab_img)

		RG_rate, RB_rate, __ 								= self.space_rates(RGB_img)
		SH_rate, __, __ 									= self.space_rates(HSV_img)

		dominant_HSV										= self.dominant_HSV_color(HSV_img)

		RG_diff, RB_diff, GB_diff							= self.space_diffs(RGB_img)

		apex_mean_R, apex_mean_G, apex_mean_B 				= self.apex_means(RGB_img)
		equator_mean_R, equator_mean_G, equator_mean_B 		= self.equator_means(RGB_img)
		stalk_mean_R, stalk_mean_G, stalk_mean_B 			= self.stalk_means(RGB_img)
		apex_mean_H, apex_mean_S, apex_mean_V 				= self.apex_means(HSV_img)
		equator_mean_H, equator_mean_S, equator_mean_V 		= self.equator_means(HSV_img)
		stalk_mean_H, stalk_mean_S, stalk_mean_V 			= self.stalk_means(HSV_img)
		apex_mean_L, apex_mean_a, apex_mean_b 				= self.apex_means(Lab_img)
		equator_mean_L, equator_mean_a, equator_mean_b 		= self.equator_means(Lab_img)
		stalk_mean_L, stalk_mean_a, stalk_mean_b 			= self.stalk_means(Lab_img)
		
		long_gradient_R, long_gradient_G, long_gradient_B 	= self.longitudinal_gradient(RGB_img)

		regions_diffs_RGB									= self.regions_mean_diffs(RGB_img)
		regions_diffs_HSV									= self.regions_mean_diffs(HSV_img)
		regions_diffs_Lab									= self.regions_mean_diffs(Lab_img)

		n_regions_means										= self.regions_mean(RGB_img, n_regions)
		n_regions_means_diffs								= self.regions_mean_diffs(RGB_img, n_regions)

		bcd  												= self.box_counting_dimension(gray_img)
		cd  												= self.correlation_dimension(gray_img)
		dd  												= self.dilation_dimension(gray_img)

		area												= self.estimated_area(gray_img)
		diameter											= self.estimated_diameter(gray_img)

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
		self.insert_feature_row(img_name, feature_values)

	def channels_mean(self, img):
		mean_0 = img[:,:,0].mean()
		mean_1 = img[:,:,1].mean()
		mean_2 = img[:,:,2].mean()

		return mean_0, mean_1, mean_2

	def space_rates(self, img):
		mean_0, mean_1, mean_2 = self.channels_mean(img)

		return mean_0/mean_1, mean_0/mean_2, mean_1/mean_2	

	def dominant_HSV_color(self, img):
		''' 0: red, 1: yellow, 2: green, 3: cyan, 4: blue, 5: magenta '''
		hue, __, __ = self.channels_mean(img)
		
		if 15 <= hue < 45:
			dominant_color = 1
		elif 45 <= hue < 75:
			dominant_color = 2
		elif 75 <= hue < 105:
			dominant_color = 3
		elif 105 <= hue < 135:
			dominant_color = 4
		elif 135 <= hue < 165:
			dominant_color = 5
		else: 
			dominant_color = 0
		
		return dominant_color

	def space_diffs(self, img):
		mean_0, mean_1, mean_2 = self.channels_mean(img)

		return mean_0 - mean_1, mean_0 - mean_2, mean_1 - mean_2

	def apex_means(self, img, OFFSET_Y_apex_stalk=0.15, HEIGHT_FRACTION=0.05):
		height, width, __ 		= img.shape

		y_c						= round(OFFSET_Y_apex_stalk * height)
		slice_height 			= round(height * HEIGHT_FRACTION / 2)
		y_f						= y_c + slice_height
		y_i						= y_c - slice_height

		x_0						= self.first_nonwhite_pixel(img, y_i, 'y')
		x_1						= self.first_nonwhite_pixel(img, y_f, 'y')
		x_i						= x_0 if x_0 > x_i else x_0

		x_0						= self.last_nonwhite_pixel(img, y_i, 'y')
		x_1						= self.last_nonwhite_pixel(img, y_f, 'y')
		x_f						= x_0 if x_0 < x_i else x_0

		mean_0, mean_1, mean_2 	= channels_mean(img[y_i:y_f, x_i:x_f])

		return mean_0, mean_1, mean_2

	def equator_means(self, img, HEIGHT_FRACTION=0.05):
		height, width, __ 		= img.shape

		y_c						= round(height / 2)
		slice_height 			= round(height * HEIGHT_FRACTION / 2)
		y_f						= y_c + slice_height
		y_i						= y_c - slice_height

		x_0						= self.first_nonwhite_pixel(img, y_i, 'y')
		x_1						= self.first_nonwhite_pixel(img, y_f, 'y')
		x_i						= x_0 if x_0 > x_i else x_0

		x_0						= self.last_nonwhite_pixel(img, y_i, 'y')
		x_1						= self.last_nonwhite_pixel(img, y_f, 'y')
		x_f						= x_0 if x_0 < x_i else x_0

		mean_0, mean_1, mean_2 	= channels_mean(img[y_i:y_f, x_i:x_f])

		return mean_0, mean_1, mean_2

	def stalk_means(self, img, OFFSET_Y_apex_stalk=0.15, HEIGHT_FRACTION=0.05):
		height, width, __ 		= img.shape

		y_c						= height - round(OFFSET_Y_apex_stalk * height)
		slice_height 			= round(height * HEIGHT_FRACTION / 2)
		y_f						= y_c + slice_height
		y_i						= y_c - slice_height
		
		x_0						= self.first_nonwhite_pixel(img, y_i, 'y')
		x_1						= self.first_nonwhite_pixel(img, y_f, 'y')
		x_i						= x_0 if x_0 > x_i else x_0

		x_0						= self.last_nonwhite_pixel(img, y_i, 'y')
		x_1						= self.last_nonwhite_pixel(img, y_f, 'y')
		x_f						= x_0 if x_0 < x_i else x_0

		mean_0, mean_1, mean_2 	= channels_mean(img[y_i:y_f, x_i:x_f])

		return mean_0, mean_1, mean_2

	def longitudinal_gradient(self, img, OFFSET_LONG=0.05, WIDTH_FRACTION=0.05):
		height, width, __ 	= img.shape

		y_i					= round(OFFSET_LONG * height)
		y_f					= height - round(OFFSET_LONG * height)

		x_c					= round(width / 2)
		slice_width			= round(width * WIDTH_FRACTION / 2)
		x_f					= x_c + slice_width
		x_i					= x_c - slice_width

		long_slice			= img[y_i:y_f,x_i:x_f,:]
		data 				= np.mean(long_slice, axis=1)

		X 		= np.arange(data.shape[0])
		X 		= pd.DataFrame(data=X, index=X)

		Y 		= [data[:,i] for i in range(3)]
		REG 	= [LinearRegression().fit(X, y) for y in Y]
		PRED 	= [reg.predict(X) for reg in REG]

		MSE 	= list()
		for i in range(3):
		    MSE.append(mean_squared_error(Y[i], PRED[i]))
		    
		ind 		= mse.index(min(MSE))
		gradient 	= REG[ind].coef_[0]

		return gradient

	def regions_mean_diffs(self, img):	
		apex_0, apex_1, apex_2 			= self.apex_means(img)
		equator_0, equator_1, equator_2	= self.equator_means(img)
		stalk_0, stalk_1, stalk_2 		= self.stalk_means(img)

		region_diffs 					= np.array([apex_0 - equator_0, equator_0 - stalk_0, apex_0 - stalk_0 \
													apex_1 - equator_1, equator_1 - stalk_1, apex_1 - stalk_1]\
													apex_2 - equator_2, equator_2 - stalk_2, apex_2 - stalk_2)

		return region_diffs

	def regions_mean(self, img, n_regions = 5):
		height, width, __ 	= img.shape

		slice_height 		= math.round(height / n_regions)

		for i in range(n_regions):
			skip 	= 1 if i > 0 else 0

			y_i 	= i * slice_height + skip
			y_f 	= y_i + slice_height

			x_0		= self.first_nonwhite_pixel(img, y_i, 'y')
			x_1		= self.first_nonwhite_pixel(img, y_f, 'y')
			x_i		= x_0 if x_0 > x_i else x_0

			x_0		= self.last_nonwhite_pixel(img, y_i, 'y')
			x_1		= self.last_nonwhite_pixel(img, y_f, 'y')
			x_f		= x_0 if x_0 < x_i else x_0

			mean_0, mean_1, mean_2 = self.channels_mean(img[y_i:y_f,x_i:x_f,:])

			if i == 0:
				means = np.array([[mean_0, mean_1, mean_2]])
			else:
				means = np.insert(means, 1, [mean_0, mean_1, mean_2], axis=0)

		return means

	def regions_mean_diffs(self, img, n_regions = 5):	
		
		means = self.region_mean(img, n_regions)

		for i in range(n_regions-1):
			for j in range(n_regions-i-1):
				if i == 0 and j == 0:
					regions_diffs = np.array([means[i]-means[j+1]]) 
				else:
					regions_diffs = np.insert(regions_diffs, 1, means[i]-means[j+1], axis=0)

		return region_diffs

	def box_counting_dimension(self, gray_img):

		def get_box_sizes(gray_img):
		    min_dim = min(gray_img.shape)
		    n 		= int(np.floor(np.log(min_dim)/np.log(2)))
		    sizes 	= 2**np.arange(n, 1, -1)
		    return sizes

		def box_count(gray_img, k):
	        img_sum = np.add.reduceat(
	            np.add.reduceat(gray_img, np.arange(0, gray_img.shape[0], k), axis=0),
	                               np.arange(0, gray_img.shape[1], k), axis=1)

	        return len(np.where((img_sum > 0) & (img_sum < 255))[0])

	    gray_img[gray_img != 255] 	= 0
	    
	    sizes 	= get_box_sizes(gray_img)
	    counts 	= []

	    for size in sizes:
	        counts.append(box_count(gray_img, size))

	    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
	    return -coeffs[0]

	def correlation_dimension(self, gray_img):
		gray_img[gray_img != 255] = 0

		height, width, __ 	= gray_img.shape
		n_pixels 			= height * width

		def heaviside_func(threshold, value):
			return int(value - threshold > 0)

		def x_ind(val):
			return int(np.ceil(val)/width)-1

		def y_ind(val):
			y_ind = val%width-1 if val%width>0 else width

		diff 		= 5
		sizes 		= np.arange(255, 0, -diff)
		sum_pixel 	= np.zeros(sizes.shape[0])

		num_it 		= 1
		for size in sizes:
			for i in range(n_pixels):
				for j in range(n_pixels):
					x_i = x_ind(i+1)
					y_i = y_ind(i+1)
					x_j = x_ind(j+1)
					y_j = y_ind(j+1)
					sum_pixel = sum_pixel + heaviside_func(size, gray_img[y_i,x_i]-gray_img[y_j,x_j])

			sum_pixel[num_it-1] = sum_pixel[num_it-1]/(n_pixels*n_pixels)
			num_it 				= num_it + 1

		coeffs = np.polyfit(np.log(sum_pixel), np.log(sizes), 1)

		return -coeffs[0]

	def dilation_dimension(self, gray_img):
		
		def invert_colors(gray_img):
			gray_img[gray_img==255] = 0
			gray_img[gray_img!=0] = 255
			return gray_img

		def get_radius_array(gray_img, h, w)
			min_dim = min(h, w)
			n = int(np.floor(np.log(min_dim)/np.log(2)))

			radius = np.arange(n, 0, -1)
			radius = radius*radius
			return radius

		def fill_image(gray_img, border, h, w):
			new_img = np.zeros((h+border*2,w+border*2), dtype='uint8')
			new_img[border:border+h,border:border+w] = gray_img
			return new_img

		def compute_sums(radius, k_size, gray_img):
			sums = []
			for r in radius:
	    		k = np.full((k_size,k_size), 0, dtype='uint8')
	    		new_k = cv2.circle(k,(int(k_size/2),int(k_size/2)), r, 1, -1)
	    		res = cv2.filter2D(gray_img, -1, new_k)
	    		sums.append(sum(sum(res)))

		h, w 	= gray_img.shape
		img 	= invert_colors(gray_img)

		radius 			= get_radius_array(gray_img, h, w)
    	diameters 		= radius*2
		max_diameter 	= max(diameters)
		
		k_size 		= max_diameter-max_diameter%100+100
		border 		= max_diameter+50
		gray_img 	= fill_image(gray_img, border, h, w)

		sums = compute_sums(radius, k_size, gray_img)

		L = [s/d for s, d in zip(sums, diameters)]

		coeffs = np.polyfit(np.log(L), np.log(1/diameters), 1)
		return -coeffs[0]

	def estimated_area(self, gray_img):
		return len(gray_img[gray_img == 255])

	def estimated_diameter(self, gray_img):
		area 		= self.estimated_area(gray_img)
		diameter 	= math.sqrt(area*4/math.pi)

		return diameter

	def first_nonwhite_pixel(self, img, known_point, axis):
		data 	= img[known_point,:,0] if axis == 'y' else img[:,known_point,0]
		i 		= np.where(data != 255)[0][0]

		return i

	def last_nonwhite_pixel(self, img, known_point, axis):
		data 	= img[known_point,:,0] if axis == 'y' else img[:,known_point,0]
		i 		= np.where(data != 255)[0][-1]

		return i