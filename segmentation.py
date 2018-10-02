import cv2
import os

path 		= os.getcwd()
folders		= ['original', 'median', 'deblurring', 'opening', 'closing', 'edge']

def otsu_thresholding(img):
	__, mask 	= cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	thr_img 	= cv2.bitwise_and(img, img, mask = mask)
	return thr_img

for folder in folders:
	files 		= os.listdir(path + '\\preprocessing\\' + folder)

	for file in files:
		img 	= cv2.imread(file, 0)
		thr_img = otsu_thresholding(img)
		cv2.imwrite(path + '\\segmentation\\' + folder + '\\' + file, thr_img)

