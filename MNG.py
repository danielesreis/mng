import cv2
from MNGPreProcessing import MNGPreProcessing
from MNGSegmentation import MNGSegmentation
from MNGContour import MNGContour
from MNGFeatures import MNGFeatures

class Mng(): 

	def __init__(self, folder, n_images):
		self.folder				= folder
		self.segmentation 		= MNGSegmentation()
		self.preprocessing 		= MNGPreProcessing(self.segmentation)
		self.contour 			= MNGContour()
		self.features 			= MNGFeatures(folder, image_names)
		self.folds 				= folds
		self.model 				= model

	def save_image(self, img_name, img, path):
		cv2.imwrite(path + img_name, img)

	def set_folds(self, folds):
		self.folds = folds

	def get_folds(self):
		return self.folds 

	def set_model(self, model):
		self.model = model

	def get_model(self):
		return self.model
	