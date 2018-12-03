import numpy as np

class MNGFeaturesRates():

	def __init__(self, features_means):
		self.features_means = features_means

	def space_rates(self, img):
		means 	= self.features_means.channels_mean(img)
		rates 	= np.array([means[0]/means[1], means[0]/means[2], means[1]/means[2]])
		return rates