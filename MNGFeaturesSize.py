import numpy as np
import math

class MNGFeaturesSize():

	def estimated_area(self, gray_img):
		return len(gray_img[gray_img == 255])

	def estimated_diameter(self, gray_img):
		area 		= self.estimated_area(gray_img)
		diameter 	= math.sqrt(area*4/math.pi)
		return diameter