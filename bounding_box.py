import cv2
import numpy as np

def get_contours(img):
	imgray 						= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	_, thresh 					= cv2.threshold(imgray, 127, 255, 0)
	image, contours, hierarchy 	= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	return contours

def get_bounding_box(img, contours):

	width, height, _ 	= img.shape
	min_x, min_y 		= width, height
	max_x = max_y 		= 0

	for contour in contours:
	    (x, y, w, h) = cv2.boundingRect(contour)
	    if x != 0 and w != width:    
	        min_x, max_x = min(x, min_x), max(x+w, max_x) 
	        
	    if y != 0 and h != height:
	        min_y, max_y = min(y, min_y), max(y+h, max_y)

	params = np.array([min_x, max_x, min_y, max_y])
	return params
