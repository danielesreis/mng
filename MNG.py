import MNG_PreProcessing
import MNG_Segmentation
import MNG_Contour
import MNG_Features
import MNG_Model

import cv2

class MNG(): 

	def __init__(self, folder, n_images):
		self.folder				= folder
		self.preprocessing 		= MNG_PreProcessing(folder)
		self.segmentation 		= MNG_Segmentation(folder)
		self.contour 			= MNG_Contour()
		self.features 			= MNG_Features(folder, image_names)
		self.model 				= MNG_Model(folder)

	def save_image(self, img_name, img, dest_folder):
		cv2.imwrite(dest.folder + img_name, img)




