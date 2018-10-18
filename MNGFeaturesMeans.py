import numpy as np

class MNGFeaturesMeans():
	
	def channels_mean(self, img):
		mean_0 	= img[:,:,0].mean()
		mean_1 	= img[:,:,1].mean()
		mean_2 	= img[:,:,2].mean()

		means 	= np.array(mean_0, mean_1, mean_2)
		return means