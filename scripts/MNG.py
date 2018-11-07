from MNGPreProcessing import MNGPreProcessing
from MNGSegmentation import MNGSegmentation
from MNGContour import MNGContour
from MNGFeatures import MNGFeatures

class MNG(): 

	def __init__(self, folder, image_names, att):
		self.folder				= folder
		self.segmentation 		= MNGSegmentation()
		self.preprocessing 		= MNGPreProcessing(self.segmentation)
		self.contour 			= MNGContour()
		self.features 			= MNGFeatures(folder, image_names, att)
		self._folds 			= None
		self._model 			= None

	def save_image(self, img_name, img, path):
		cv2.imwrite(path + img_name, img)

	@property
	def folds(self):
		return self._folds 

	@folds.setter
	def folds(self, folds):
		self._folds = folds

	@property
	def model(self):
		return self._model

	@model.setter
	def model(self, model):
		self._model = model

	