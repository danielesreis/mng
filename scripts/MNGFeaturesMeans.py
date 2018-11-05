import numpy as np

class MNGFeaturesMeans():
	
	def channels_mean(self, img):
		chn_0 	= img[:,:,0]
		chn_1 	= img[:,:,1]
		chn_2 	= img[:,:,2]

		nonwhite_0 	= chn_0[chn_0 != 255] 
		nonwhite_1 	= chn_1[chn_1 != 255] 
		nonwhite_2 	= chn_2[chn_2 != 255] 

		mean_0 = np.sum(nonwhite_0)/len(nonwhite_0)
		mean_1 = np.sum(nonwhite_1)/len(nonwhite_1)
		mean_2 = np.sum(nonwhite_2)/len(nonwhite_2)

		means 	= np.array([mean_0, mean_1, mean_2])
		return means