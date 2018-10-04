import cv2
import os
from bounding_box import get_contours, get_bounding_box
from preprocessing import remove_bg_noise

path 		= os.getcwd()
folders		= ['original', 'median', 'deblurring', 'opening', 'closing', 'sharpening']

def get_mask(grayscale_img):
	__, mask 	= cv2.threshold(grayscale_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	inv_mask	= cv2.bitwise_not(mask)

	return mask

def get_inverse_mask(grayscale_img):
	mask = get_mask(grayscale_img)
	inv_mask = cv2.bitwise_not(mask)

	return inv_mask

def convert_to_mask(img):
	mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	mask = cv2.bitwise_not(mask)

	return mask

def otsu_thresholding(color_img):
	grayscale_img 			= cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
	mask 					= get_mask(grayscale_img)
	bg 						= remove_bg_noise(color_img, mask)
	new_mask 				= convert_to_mask(bg)
	thr_img 				= cv2.bitwise_and(color_img, color_img, mask = new_mask)
	thr_img[thr_img == 0] 	= 255
	
	return thr_img

def main():
	for folder in folders:
		files 		= os.listdir(path + '\\preprocessing\\' + folder)

		for file in files:
			# bgr
			color_img 		= cv2.imread(file)
			thr_img 		= otsu_thresholding(color_img)

			contours 		= get_contours(clean_img)
			box 			= get_bounding_box(clean_img, contours)
			res_img 		= clean_img[box[0]:box[1], box[2]:box[3]]
			# rotate
			cv2.imwrite(path + '\\segmentation\\' + folder + '\\' + file, res_img)

main()

