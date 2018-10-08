import numpy as np
import cv2
import os
from _unsharp_mask import unsharp_mask

path 			= os.getcwd() 
files 			= os.listdir(path + '\\images')

# test these
MF_WINDOW 		= 15
KERNEL 			= np.ones((5,5), np.uint8)
RADIUS 			= 1
AMOUNT 			= 1

# def remove_bg_noise(color_img, mask):
# 	bg 			= cv2.bitwise_and(color_img, color_img, mask = mask)
# 	bg[bg != 0] = 255
# 	return bg

# def remove_shadow(img):

def main():

	for file in files:
		img 		= cv2.imread(file)
		dest_folder = path + '\\preprocessing\\'

		img = remove_shadow(img)

		cv2.imwrite(dest_folder + 'original\\'		+ file, img)

		filt_img = cv2.medianBlur(img, MF_WINDOW)
		cv2.imwrite(dest_folder + 'median\\' 		+ file, filt_img)


		cv2.imwrite(dest_folder + 'deblurring\\' 	+ file, filt_img)

		filt_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, KERNEL)
		cv2.imwrite(dest_folder + 'opening\\' 		+ file, filt_img)

		filt_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, KERNEL)
		cv2.imwrite(dest_folder + 'closing\\' 		+ file, filt_img)

		filt_img = unsharp_mask(img, radius = RADIUS, amount = AMOUNT)
		cv2.imwrite(dest_folder + 'sharpening\\'	+ file, filt_img)

main()

