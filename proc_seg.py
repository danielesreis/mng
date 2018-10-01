import numpy as np
import cv2
import os

path 			= os.getcwd() + '\\images'
files 			= os.listdir(path)

original_p 		= path + '\\..\\preprocessing\\original'
median_p 		= path + '\\..\\preprocessing\\median_filter'
deblurring_p	= path + '\\..\\preprocessing\\deblurring_filter'
opening_p		= path + '\\..\\preprocessing\\opening_operation'
closing_p		= path + '\\..\\preprocessing\\closing_operation'
edge_p			= path + '\\..\\preprocessing\\edge_sharpening'

original_t 		= path + '\\..\\segmentation\\original'
median_t 		= path + '\\..\\segmentation\\median'
deblurring_t 	= path + '\\..\\segmentation\\deblurring'
opening_t 		= path + '\\..\\segmentation\\opening'
closing_t 		= path + '\\..\\segmentation\\closing'
edge_t  		= path + '\\..\\segmentation\\edge'

# test these
MF_WINDOW 		= 5
KERNEL 			= np.ones((5,5), np.uint8)

for file in files:
	img = cv2.imread(file)

	cv2.imwrite(original_p		+ file, img)
	__,thr_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	cv2.imwrite(original_t		+ file, thr_img)

	filt_img = cv2.medianBlur(img, MF_WINDOW)
	cv2.imwrite(median_p 		+ file, filt_img)
	__,thr_img = cv2.threshold(filt_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	cv2.imwrite(median_t		+ file, thr_img)


	cv2.imwrite(deblurring_p 	+ file, filt_img)
	__,thr_img = cv2.threshold(filt_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	cv2.imwrite(deblurring_t	+ file, thr_img)

	filt_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, KERNEL)
	cv2.imwrite(opening_p 		+ file, filt_img)
	__,thr_img = cv2.threshold(filt_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	cv2.imwrite(opening_t		+ file, thr_img)

	filt_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, KERNEL)
	cv2.imwrite(closing_p 		+ file, filt_img)
	__,thr_img = cv2.threshold(filt_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	cv2.imwrite(closing_t		+ file, thr_img)


	cv2.imwrite(edge_p 			+ file, filt_img)
	__,thr_img = cv2.threshold(filt_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	cv2.imwrite(edge_t			+ file, thr_img)



