import numpy as np
import cv2
from _unsharp_mask import unsharp_mask

class MNG_PreProcessing():

	def __init__(self, folder):
		self.dest_folder = folder + 'preprocessing\\'

	# def rotate(img):

	def remove_shadow(img):
		# this may change according to the mango
		filt_img = simple_thresholding(img, 100)

		return filt_img

	def median_filter(img, MF_WINDOW=15):
		filt_img = cv2.medianBlur(img, MF_WINDOW)

		return filt_img

	# def deblurring(img):

	def opening_operation(img, KERNEL=np.ones((5,5), np.uint8)):
		filt_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, KERNEL)
		
		return filt_img

	def closing_operation(img, KERNEL=np.ones((5,5), np.uint8)):
		filt_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, KERNEL)
		
		return filt_img

	def unsharp_masking(img, RADIUS=1, AMOUNT=1):
		filt_img = unsharp_mask(img, radius=RADIUS, amount=AMOUNT)
		
		return filt_img