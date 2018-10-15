import MNG_PreProcessing
import MNG_Segmentation
import MNG_Contour
import MNG_Features
import MNG_Model

import cv2

class MNG(): 

	def __init__(self, folder):
		self.folder				= folder
		self.preprocessing 		= MNG_PreProcessing(folder)
		self.segmentation 		= MNG_Segmentation(folder)
		self.contour 			= MNG_Contour()
		self.features 			= MNG_Features(folder)
		self.model 				= MNG_Model(folder)

	def save_image(img_name, img, folder):
		cv2.imwrite(folder + img_name, img)




