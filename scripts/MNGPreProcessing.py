import numpy as np
import cv2
import matplotlib.pyplot as plt
# from _unsharp_mask import unsharp_mask

class MNGPreProcessing():

	def __init__(self, segmentation):
		self.segmentation = segmentation

	def remove_shadow(self, color_img, threshold=100):
		filt_img = self.segmentation.simple_thresholding(color_img, threshold)
		return filt_img

	def median_filter(self, color_img, MF_WINDOW=15):
		filt_img = cv2.medianBlur(color_img, MF_WINDOW)
		return filt_img

	def opening_operation(self, img, KERNEL=np.ones((5,5),np.uint8)):
		__, mask = self.segmentation.otsu_thresholding(img)
		filt_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
		filt_img = cv2.bitwise_and(img, img, mask=filt_mask)
		return filt_img

	def closing_operation(self, img_holes, img_no_holes, KERNEL=np.ones((5,5),np.uint8)):
		__, mask = self.segmentation.otsu_thresholding(img_holes)
		filt_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
		filt_img = cv2.bitwise_and(img_no_holes, img_no_holes, mask=filt_mask)
		return filt_img

	def unsharp_masking(self, color_img, RADIUS=1, AMOUNT=1):
		filt_img = unsharp_mask(color_img, radius=RADIUS, amount=AMOUNT, multichannel=True, preserve_range=True)
		return filt_img