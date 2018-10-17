import numpy as np

class MNGFeaturesRates():

	def __init__(self, features_means):
		self.features_means = features_means

	def space_rates(self, img):
		mean_0, mean_1, mean_2 	= self.features_means.channels_mean(img)
		rates 					= np.array([mean_0/mean_1, mean_0/mean_2, mean_1/mean_2])
		return rates