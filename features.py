import cv2
import math

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


def equator_means(img):


def stalk_means(img):


def longitudinal_gradient(img):


def regions_mean_diffs(img):


def box_counting_dimension(img):


def correlation_dimension(img):


def dilation_dimension(img):


def estimated_area(binary_img):
	return len(binary_img[binary_img == 255])

def estimated_diameter(binary_img):
	area = estimated_area(binary_img)
	diameter = math.sqrt(area * 4 / math.pi)
