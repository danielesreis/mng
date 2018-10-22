import pandas as pd
import os

class MNGFolds():

	def __init__(self, folder, features_path, k):
		self.origin_folder 	= folder
		self.dest_folder 	= folder + '..\\features\\'
		self.features_path	= features_path
		self.k 				= k
		
		self.set_data()
		self.set_n()
		self.set_folders_names()

	def set_data(self):
		self.data = pd.read_csv(self.features_path, sep=';', index_col=0)

	def set_n(self):
		self.n = len(self.data)

	def set_folders_names(self):
		# palmer_folders 	= ['palmer1', 'palmer2', 'palmer3', 'palmer4', 'palmer5', 'palmer6', 'palmer7']
		# tommy_folders 	= ['tommy1', 'tommy2', 'tommy3', 'tommy4', 'tommy5', 'tommy6', 'tommy7']

	def get_fold_data(self, fold_number):
		
		return train, test

	def separate_folds(self):

		palmer_data = self.data.loc['palmer':'tommy']
		tommy_data 	= self.data.loc['tommy':]

		palmer_data.to_csv(self.dest_folder + 'palmer_all.csv', sep=';')
		tommy_data.to_csv(self.dest_folder + 'tommy_all.csv', sep=';')

		n_palmer 	= int(len(palmer_data)/self.k)
		n_tommy 	= int(len(tommy_data)/self.k)

		for i in range(self.k-1):
			ind1 = i*n_palmer
			ind2 = (i+1)*n_palmer
			palmer_fold = palmer_data.iloc[ind1:ind2]
			palmer_fold.to_csv(self.dest_folder + self.palmer_folders[i] + '.csv', sep=';')

			ind1 = i*n_tommy
			ind2 = (i+1)*n_tommy
			tommy_fold = tommy_data.iloc[ind1:ind2]
			tommy_fold.to_csv(self.dest_folder + self.tommy_folders[i] + '.csv', sep=';')


		palmer_fold = palmer_data.iloc[n_palmer*(self.k-1):]
		palmer_fold.to_csv(self.dest_folder + self.palmer_folders[-1] + '.csv', sep=';')
		
		tommy_fold 	= tommy_data.iloc[n_tommy*(self.k-1):]
		tommy_fold.to_csv(self.dest_folder + self.tommy_folders[-1] + '.csv', sep=';')



