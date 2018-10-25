import numpy as np
import cv2
from _unsharp_mask import unsharp_mask

class MNGPreProcessing():

	def __init__(self, segmentation):
		self.segmentation = segmentation

	def remove_shadow(self, color_img):
		# this may change according to the mango
		filt_img = self.segmentation.simple_thresholding(color_img, 100)
		return filt_img

	def median_filter(self, color_img, MF_WINDOW=15):
		filt_img = cv2.medianBlur(color_img, MF_WINDOW)
		return filt_img

	def opening_operation(self, color_img, KERNEL=np.ones((5,5), np.uint8)):
		filt_img = cv2.morphologyEx(color_img, cv2.MORPH_OPEN, KERNEL)
		return filt_img

	def closing_operation(self, color_img, KERNEL=np.ones((5,5), np.uint8)):
		filt_img = cv2.morphologyEx(color_img, cv2.MORPH_CLOSE, KERNEL)
		return filt_img

	def unsharp_masking(self, color_img, RADIUS=1, AMOUNT=1):
		filt_img = unsharp_mask(color_img, radius=RADIUS, amount=AMOUNT, multichannel=True)
		return filt_img