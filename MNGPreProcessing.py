import numpy as np
import cv2
from _unsharp_mask import unsharp_mask

class MNGPreProcessing():

	def __init__(self, segmentation):
		self.segmentation = segmentation

	def remove_shadow(self, img):
		# this may change according to the mango
		filt_img = self.segmentation.simple_thresholding(img, 100)

		return filt_img

	def median_filter(self, img, MF_WINDOW=15):
		filt_img = cv2.medianBlur(img, MF_WINDOW)

		return filt_img

	# def deblurring(self, img):

	def opening_operation(self, img, KERNEL=np.ones((5,5), np.uint8)):
		filt_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, KERNEL)
		
		return filt_img

	def closing_operation(self, img, KERNEL=np.ones((5,5), np.uint8)):
		filt_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, KERNEL)
		
		return filt_img

	def unsharp_masking(self, img, RADIUS=1, AMOUNT=1):
		filt_img = unsharp_mask(img, radius=RADIUS, amount=AMOUNT)
		
		return filt_img