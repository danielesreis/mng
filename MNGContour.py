import cv2
import numpy as np

class MNGContour():

	
	def get_contours(self, color_img):
		imgray 				= cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
		__, thresh 			= cv2.threshold(imgray, 127, 255, 0)
		__, contours, __	= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		return contours

	def get_bounding_box(self, img, contours):
		img_h, img_w, __ = img.shape

		for contour in countours:
			(x,y,w,h) = cv2.boundingRect(contour)
			
			height_cond = h > 1500 and is not img_h
			width_cond = w > 800 and is not img_w
			if height_cond and width_cond:
				box_contour = contour
				break

		min_x 			= x
		max_x 			= x + w
		min_y 			= y
		max_y 			= y + h

		box 			= np.array([min_x, max_x, min_y, max_y])
		return box
		
	def cut_image(self, color_img):
		contours 	= get_contours(color_img)
		box 		= get_bounding_box(color_img, contours)
		res_img 	= color_img[box[2]:box[3], box[0]:box[1]]

		return res_img
