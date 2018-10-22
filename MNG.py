import cv2
import MNGPreProcessing
import MNGSegmentation
import MNGContour
import MNGFeatures
import MNGFolds
import MNGModel


class MNG(): 

	def __init__(self, folder, n_images):
		self.folder				= folder
		self.segmentation 		= MNGSegmentation()
		self.preprocessing 		= MNGPreProcessing(self.segmentation)
		self.contour 			= MNGContour()
		self.features 			= MNGFeatures(folder, image_names)
		self.folds 				= folds
		self.model 				= model

	def save_image(self, img_name, img, dest_folder):
		cv2.imwrite(dest.folder + img_name, img)

	@set_folds.setter
	def set_folds(self, folds):
		self.folds = folds

	@property
	def get_folds(self):
		return self.folds 

	@set_model.setter
	def set_model(self, model):
		self.model = model

	@property
	def get_model(self):
		return self.model
	



