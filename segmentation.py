import cv2
import os
from contours import get_contours, get_bounding_box
from thresholding import otsu_thresholding
from preprocessing import remove_shadow

path 		= os.getcwd()
folders		= ['original', 'median', 'deblurring', 'opening', 'closing', 'sharpening']

def main():
	for folder in folders:
		files 		= os.listdir(path + '\\preprocessing\\' + folder)

		for file in files:
			# bgr
			color_img 		= cv2.imread(file)
			thr_img 		= otsu_thresholding(color_img)
			filtered_img	= remove_shadow(thr_img)

			contours 		= get_contours(filtered_img)
			box 			= get_bounding_box(filtered_img, contours)
			res_img 		= filtered_img[box[2]:box[3], box[0]:box[1]]
			# rotate
			cv2.imwrite(path + '\\segmentation\\' + folder + '\\' + file, res_img)

main()

