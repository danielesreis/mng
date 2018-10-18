import numpy as np

class MNGFeaturesRates():

	def __init__(self, features_means):
		self.features_means = features_means

	def space_rates(self, img):
		means 	= self.features_means.channels_mean(img)
		rates 	= np.array([mean[0]/mean[1], mean[0]/mean[2], mean[1]/mean[2]])
		return rates