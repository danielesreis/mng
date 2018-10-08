import cv2
import numpy as np

def get_contours(color_img):
	imgray 				= cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
	__, thresh 			= cv2.threshold(imgray, 127, 255, 0)
	__, contours, __	= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	return contours

def get_bounding_box(img, contours):

	# This may change accordingly to preprocessing
	# Consider changing it to something more sophisticated, e.g.: checking width and height and choosing an appropriate contour
	contour 		= contours[54]
	(x, y, w, h) 	= cv2.boundingRect(contour)

	min_x 			= x
	max_x 			= x + w
	min_y 			= y
	max_y 			= y + h

	box 			= np.array([min_x, max_x, min_y, max_y])
	return box
