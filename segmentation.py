import cv2
import os
from bounding_box import get_contours, get_bounding_box

path 		= os.getcwd()
folders		= ['original', 'median', 'deblurring', 'opening', 'closing', 'sharpening']

def otsu_thresholding(grayscale_img, color_img):
	__, mask 	= cv2.threshold(grayscale_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	inv_mask	= cv2.bitwise_not(mask)
	thr_img 	= cv2.bitwise_and(color_img, color_img, mask = inv_mask)
	return thr_img

for folder in folders:
	files 		= os.listdir(path + '\\preprocessing\\' + folder)

	for file in files:
		# bgr
		color_img 		= cv2.imread(file)
		grayscale_img 	= cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
		thr_img 		= otsu_thresholding(grayscale_img, img)
		contours 		= get_contours(thr_img)
		box 			= get_bounding_box(thr_img, contours)
		# cut img 
		cv2.imwrite(path + '\\segmentation\\' + folder + '\\' + file, thr_img)

