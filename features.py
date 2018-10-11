import cv2
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

OFFSET_long 			= 0.05
OFFSET_Y_apex_stalk 	= 0.15
OFFSET_X_apex_stalk 	= 0.05

HEIGHT_FRACTION			= 0.05
WIDTH_FRACTION			= 0.05

def channels_mean(img):
	mean_0 = img[:,:,0].mean()
	mean_1 = img[:,:,1].mean()
	mean_2 = img[:,:,2].mean()

	return mean_0, mean_1, mean_2

def space_rates(img):
	mean_0, mean_1, mean_2 = channels_mean(img)

	return mean_0/mean_1, mean_0/mean_2, mean_1/mean_2	

def dominant_HSV_color(img):
	''' 0: red, 1: yellow, 2: green, 3: cyan, 4: blue, 5: magenta '''
	hue, __, __ = channels_mean(img)
	
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

def space_diffs(img):
	mean_0, mean_1, mean_2 = channels_mean(img)

	return mean_0 - mean_1, mean_0 - mean_2, mean_1 - mean_2

def apex_means(img):
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

	mean_0, mean_1, mean_2 	= channels_mean(img[y_i:y_f, x_i:x_f])

	return mean_0, mean_1, mean_2

def equator_means(img):
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

	mean_0, mean_1, mean_2 	= channels_mean(img[y_i:y_f, x_i:x_f])

	return mean_0, mean_1, mean_2

def stalk_means(img):
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

	mean_0, mean_1, mean_2 	= channels_mean(img[y_i:y_f, x_i:x_f])

	return mean_0, mean_1, mean_2

def longitudinal_gradient(img):
	height, width, __ 	= img.shape

	y_i					= round(OFFSET_long * height)
	y_f					= height - round(OFFSET_long * height)

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

def regions_mean_diffs(img):	
	apex_0, apex_1, apex_2 			= apex_means(img)
	equator_0, equator_1, equator_2	= equator_means(img)
	stalk_0, stalk_1, stalk_2 		= stalk_means(img)

	region_diffs 					= np.array([apex_0 - equator_0, equator_0 - stalk_0, apex_0 - stalk_0 \
												apex_1 - equator_1, equator_1 - stalk_1, apex_1 - stalk_1]\
												apex_2 - equator_2, equator_2 - stalk_2, apex_2 - stalk_2)

	return region_diffs

def regions_mean(img, n_regions):
	height, width, __ 	= img.shape

	slice_height 		= math.round(height / n_regions)

	for i in range(n_regions):
		skip 	= 1 if i > 0 else 0

		y_i 	= i * slice_height + skip
		y_f 	= y_i + slice_height

		x_0		= first_nonwhite_pixel(img, y_i, 'y')
		x_1		= first_nonwhite_pixel(img, y_f, 'y')
		x_i		= x_0 if x_0 > x_i else x_0

		x_0		= last_nonwhite_pixel(img, y_i, 'y')
		x_1		= last_nonwhite_pixel(img, y_f, 'y')
		x_f		= x_0 if x_0 < x_i else x_0

		mean_0, mean_1, mean_2 = channels_mean(img[y_i:y_f,x_i:x_f,:])

		if i == 0:
			means = np.array([[mean_0, mean_1, mean_2]])
		else:
			means = np.insert(means, 1, [mean_0, mean_1, mean_2], axis=0)

	return means

def regions_mean_diffs(img, n_regions):	
	
	means = region_mean(img, n_regions)

	for i in range(n_regions-1):
		for j in range(n_regions-i-1):
			if i == 0 and j == 0:
				regions_diffs = np.array([means[i]-means[j+1]]) 
			else:
				regions_diffs = np.insert(regions_diffs, 1, means[i]-means[j+1], axis=0)

	return region_diffs

def box_counting_dimension(grayscale_img):

	def box_count(grayscale_img, k):
        img_sum = np.add.reduceat(
            np.add.reduceat(grayscale_img, np.arange(0, grayscale_img.shape[0], k), axis=0),
                               np.arange(0, grayscale_img.shape[1], k), axis=1)

        return len(np.where((img_sum > 0) & (img_sum < 255))[0])

    grayscale_img[grayscale_img != 255] = 0

    # min_dim 	= min(grayscale_img.shape)
    # n 			= 2**np.floor(np.log(min_dim)/np.log(2))
    # n 			= int(np.log(n)/np.log(2))
    n 			= 20

    sizes 		= 2**np.arange(n, 1, -1)
    counts 		= []
    for size in sizes:
        counts.append(box_count(grayscale_img, size))

    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def correlation_dimension(grayscale_img):
	grayscale_img[grayscale_img != 255] = 0

	height, width, __ 	= grayscale_img.shape
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
				sum_pixel = sum_pixel + heaviside_func(size, grayscale_img[y_i,x_i]-grayscale_img[y_j,x_j])

		sum_pixel[num_it-1] = sum_pixel[num_it-1]/(n_pixels*n_pixels)
		num_it 				= num_it + 1

	coeffs = np.polyfit(np.log(sum_pixel), np.log(sizes), 1)

	return -coeffs[0]

def dilation_dimension(grayscale_img):
	grayscale_img[grayscale_img != 255] = 0

	kernel = np.array([	[0, 0, 0, 0, 1, 0, 0, 0, 0], \
						[0, 0, 0, 1, 1, 1, 0, 0, 0], \
						[0, 0, 1, 1, 1, 1, 1, 0, 0], \
						[0, 0, 0, 1, 1, 1, 0, 0, 0], \
						[0, 0, 0, 0, 1, 0, 0, 0, 0]])

	# kernels = 
	# diameters = 
	results	= [cv2.filter2D(grayscale_img, -1, kernel) for kernel in kernels]
	sums 	= [sum(sum(result)) for result in results]
	L 		= [summ/diameter for summ, diameter in zip(sums, diameters)]

	coeffs = np.polyfit(np.log(L), np.log(1/diameters), 1)

	return -coeffs[0]


def estimated_area(grayscale_img):
	return len(grayscale_img[grayscale_img == 255])

def estimated_diameter(grayscale_img):
	area 		= estimated_area(grayscale_img)
	diameter 	= math.sqrt(area * 4 / math.pi)

	return diameter

def first_nonwhite_pixel(img, known_point, axis):
	data 	= img[known_point,:,0] if axis == 'y' else img[:,known_point,0]
	i 		= np.where(data != 255)

	return i[0][0]

def last_nonwhite_pixel(img, known_point, axis):
	data 	= img[known_point,:,0] if axis == 'y' else img[:,known_point,0]
	i 		= np.where(data != 255)

	return i[0][-1]