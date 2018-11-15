import numpy as np
import math

class MNGFeaturesSize():

	def estimated_area(self, gray_img):
		return len(gray_img[gray_img != 255])

	def estimated_measures(self, gray_img):
		h, w = gray_img.shape

		axis_0 		= gray_img.sum(axis=0, dtype='int32')
		axis_0_px 	= np.where(axis_0 != 255*h)

		axis_1 		= gray_img.sum(axis=1, dtype='int32')
		axis_1_px 	= np.where(axis_1 != 255*w)

		if h > w:
			width 	= axis_0_px[0][-1] - axis_0_px[0][0]
			height 	= axis_1_px[0][-1] - axis_1_px[0][0]
		else:
			height 	= axis_0_px[0][-1] - axis_0_px[0][0]
			width 	= axis_1_px[0][-1] - axis_1_px[0][0]

		return height, width

	def estimated_diameter(self, gray_img):
		area 		= self.estimated_area(gray_img)
		diameter 	= math.sqrt(area*4/math.pi)
		return diameter