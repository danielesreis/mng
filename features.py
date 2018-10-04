import cv2
import math
import numpy as np

OFFSET_long 			= 0.05
OFFSET_Y_apex_stalk 	= 0.15
OFFSET_X_apex_stalk 	= 0.05

SCALAR_apex_stalk	= 0.05

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
	width, height, _ 		= img.shape

	y_i, _ 					= height - round(OFFSET_Y_apex_stalk * height)
	slice_height 			= round(height * SCALAR_apex_stalk)
	y_f						= y_i - slice_height

	# get first non zero pixel
	x_i 					= round(OFFSET_X_apex_stalk * width)
	x_f						= width - x_i

	mean_0, mean_1, mean_2 	= channels_mean(img[x_i:x_f, y_i:y_f])

	return mean_0, mean_1, mean_2

def equator_means(img):
	width, height, _ 		= img.shape

	y_i, _ 					= height - round(0.5 * height)
	slice_height 			= round(height * SCALAR_apex_stalk)
	y_f						= y_i - slice_height

	x_i 					= round(OFFSET_X_apex_stalk * width)
	x_f						= width - x_i

	mean_0, mean_1, mean_2 	= channels_mean(img[x_i:x_f, y_i:y_f])	

def stalk_means(img):
	width, height, _ 		= img.shape

	y_i, _ 					= round(OFFSET_Y_apex_stalk * height)
	slice_height 			= round(height * SCALAR_apex_stalk)
	y_f						= y_i + slice_height

	# get first non zero pixel
	x_i 					= round(OFFSET_X_apex_stalk * width)
	x_f						= width - x_i

	mean_0, mean_1, mean_2 	= channels_mean(img[x_i:x_f, y_i:y_f])	

	return mean_0, mean_1, mean_2

def longitudinal_gradient(img):
	grayscale_img 		= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	width, height, _ 	= img.shape

	x_i, x_f			= get_hor_points(img, width)
	y_i, y_f			= get_vert_points(img, height, OFFSET_long)


def regions_mean_diffs(img):	
	apex_R, apex_G, apex_B 			= apex_means(img)
	equator_R, equator_G, equator_B	= equator_means(img)
	stalk_R, stalk_G, stalk_B 		= stalk_means(img)

	region_diffs 					= np.array([apex_R - equator_R, equator_R - stalk_R, apex_R - stalk_R \
												apex_G - equator_G, equator_G - stalk_G, apex_G - stalk_G]\
												apex_B - equator_B, equator_B - stalk_B, apex_B - stalk_B)

	return region_diffs

def box_counting_dimension(img):


def correlation_dimension(img):


def dilation_dimension(img):


def estimated_area(binary_img):
	return len(binary_img[binary_img == 255])

def estimated_diameter(binary_img):
	area 		= estimated_area(binary_img)
	diameter 	= math.sqrt(area * 4 / math.pi)

	return diameter

def get_hor_points(img, width):
	middle_width 	= OFFSET_long * width
	hor_offset 		= round((width - middle_width) / 2)

	x_i 			= hor_offset
	x_f 			= x_i + middle_width

	return x_i, x_f

def get_vert_points(img, height, offset):
	vert_offset		= round(offset * height)

	y_i				= height - vert_offset
	y_f				= vert_offset

	return y_i, y_f
