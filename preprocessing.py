import numpy as np
import cv2
import os

path 			= os.getcwd() 
files 			= os.listdir(path + '\\images')

# test these
MF_WINDOW 		= 5
KERNEL 			= np.ones((5,5), np.uint8)

def main():

	for file in files:
		img 		= cv2.imread(file)
		dest_folder = path + '\\preprocessing\\'

		cv2.imwrite(dest_folder + 'original\\'		+ file, img)

		filt_img = cv2.medianBlur(img, MF_WINDOW)
		cv2.imwrite(dest_folder + 'median\\' 		+ file, filt_img)


		cv2.imwrite(dest_folder + 'deblurring\\' 	+ file, filt_img)

		filt_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, KERNEL)
		cv2.imwrite(dest_folder + 'opening\\' 		+ file, filt_img)

		filt_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, KERNEL)
		cv2.imwrite(dest_folder + 'closing\\' 		+ file, filt_img)


		cv2.imwrite(dest_folder + 'edge\\' 			+ file, filt_img)

main()

