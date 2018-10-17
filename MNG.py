import MNGPreProcessing
import MNGSegmentation
import MNGContour
import MNGFeatures
import MNGModel

import cv2

class MNG(): 

	def __init__(self, folder, n_images):
		self.folder				= folder
		self.segmentation 		= MNGSegmentation()
		self.preprocessing 		= MNGPreProcessing(self.segmentation)
		self.contour 			= MNGContour()
		self.features 			= MNGFeatures(folder, image_names)
		self.model 				= MNGModel(folder)

	def save_image(self, img_name, img, dest_folder):
		cv2.imwrite(dest.folder + img_name, img)




