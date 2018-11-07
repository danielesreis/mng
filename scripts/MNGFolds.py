import numpy as np
import pandas as pd
import os

class MNGFolds():

	def __init__(self, folder, features_path, k, n=60):
		self.dest_folder	 	= folder + '..\\features\\'
		self.features_path		= features_path
		self.k	 				= k
		self.data	 			= self.set_data(features_path)
		self.n	 				= n
		self.palmer_folders,\
		self.tommy_folders 		= self.set_folders_names(k)

	def set_data(self, features_path):
		return pd.read_csv(features_path, sep=';', index_col=0)

	def set_folders_names(self, k):

		palmer_names 	= ['palmer_sem']*self.k
		tommy_names 	= ['tommy_sem']*self.k

		nums = list(np.arange(1, self.k+1))

		palmer_folders 	= [palmer+str(num) for palmer,num in zip(palmer_names,nums)]
		tommy_folders 	= [tommy+str(num) for tommy,num in zip(tommy_names,nums)]

		return palmer_folders, tommy_folders

	def get_fold_data(self, fold, model_type):

		if model_type is 'all':
			train 	= pd.read_csv(self.dest_folder + 'datasets\\all\\' + str(fold) + '\\train_sem_' + str(fold) + '.csv', sep=';', index_col=0)
			test 	= pd.read_csv(self.dest_folder + 'datasets\\all\\' + str(fold) + '\\teste_' + str(fold) + '.csv', sep=';', index_col=0)
		elif model_type is 'palmer':
			train 	= pd.read_csv(self.dest_folder + 'datasets\\palmer\\' + str(fold) + '\\train_sem_' + str(fold) + '.csv', sep=';', index_col=0)
			test 	= pd.read_csv(self.dest_folder + 'datasets\\palmer\\' + str(fold) + '\\teste_' + str(fold) + '.csv', sep=';', index_col=0)
		else:
			train 	= pd.read_csv(self.dest_folder + 'datasets\\tommy\\' + str(fold) + '\\train_sem_' + str(fold) + '.csv', sep=';', index_col=0)
			test 	= pd.read_csv(self.dest_folder + 'datasets\\tommy\\' + str(fold) + '\\teste_' + str(fold) + '.csv', sep=';', index_col=0)
				
		return train, test

	def concat_folds(self):

		nums = np.arange(1, self.k+1)

		for fold in range(self.k):
			palmer_test 	= pd.read_csv(self.dest_folder + 'palmer\\fold' + str(fold+1) + '.csv', sep=';', index_col=0)
			tommy_test 		= pd.read_csv(self.dest_folder + 'tommy\\fold' + str(fold+1) + '.csv', sep=';', index_col=0)
			all_test		= pd.concat([palmer_test, tommy_test])

			palmer_test.to_csv(self.dest_folder + 'datasets\\palmer\\' + str(fold+1) + '\\teste_' + str(fold+1) + '.csv', sep=';')
			tommy_test.to_csv(self.dest_folder + 'datasets\\tommy\\' + str(fold+1) + '\\teste_' + str(fold+1) + '.csv', sep=';')
			all_test.to_csv(self.dest_folder + 'datasets\\all\\' + str(fold+1) + '\\teste_' + str(fold+1) + '.csv', sep=';')

			train_folds 	= np.setdiff1d(nums, fold+1)
			palmer_train 	= pd.DataFrame([])
			tommy_train 	= pd.DataFrame([])
			all_train	 	= pd.DataFrame([])

			for train_fold in train_folds:
				palmer 	= pd.read_csv(self.dest_folder + 'palmer\\fold' + str(train_fold) + '.csv', sep=';', index_col=0)
				tommy 	= pd.read_csv(self.dest_folder + 'tommy\\fold' + str(train_fold) + '.csv', sep=';', index_col=0)

				palmer_train 	= pd.concat([palmer_train, palmer])
				tommy_train 	= pd.concat([tommy_train, tommy])
				all_train 		= pd.concat([all_train, palmer, tommy])

			palmer_train.to_csv(self.dest_folder + 'datasets\\palmer\\' + str(fold+1) + '\\train_sem_' + str(fold+1) + '.csv', sep=';')
			tommy_train.to_csv(self.dest_folder + 'datasets\\tommy\\' + str(fold+1) + '\\train_sem_' + str(fold+1) + '.csv', sep=';')
			all_train.to_csv(self.dest_folder + 'datasets\\all\\' + str(fold+1) + '\\train_sem_' + str(fold+1) + '.csv', sep=';')

	def organize_folds(self):
		# must be an even number	
		n_fold = int(self.n/self.k)

		for fold in range(self.k):

			palmer 		= pd.DataFrame([])
			tommy 		= pd.DataFrame([])
			all_data 	= pd.DataFrame([])

			ind1 = fold*n_fold 
			ind2 = ind1+n_fold

			for i in range(self.k):

				palmer_i_fold 	= pd.read_csv(self.dest_folder + 'palmer\\' + self.palmer_folders[i] + '.csv', sep=';', index_col=0)
				tommy_i_fold 	= pd.read_csv(self.dest_folder + 'tommy\\' + self.tommy_folders[i] + '.csv', sep=';', index_col=0)

				if fold == self.k-1:
					palmer_slice 	= palmer_i_fold.iloc[ind1:]
					tommy_slice		= tommy_i_fold.iloc[ind1:]
					
				else:
					palmer_slice 	= palmer_i_fold.iloc[ind1:ind2]
					tommy_slice 	= tommy_i_fold.iloc[ind1:ind2]

				palmer 		= pd.concat([palmer, palmer_slice])
				tommy 		= pd.concat([tommy, tommy_slice])
				all_data	= pd.concat([all_data, palmer_slice, tommy_slice])

			palmer.to_csv(self.dest_folder + 'palmer\\fold' + str(fold+1) + '.csv', sep=';')
			tommy.to_csv(self.dest_folder + 'tommy\\fold' + str(fold+1) + '.csv', sep=';')
			all_data.to_csv(self.dest_folder + 'all\\all_fold' + str(fold+1) + '.csv', sep=';')

		self.concat_folds()

	def separate_folds(self):
		palmer_cond = self.data.index.str.contains('palmer')
		tommy_cond 	= self.data.index.str.contains('tommy')

		palmer_data = self.data.loc[palmer_cond]
		tommy_data 	= self.data.loc[tommy_cond]

		palmer_data.to_csv(self.dest_folder + 'palmer\\palmer_all.csv', sep=';')
		tommy_data.to_csv(self.dest_folder + 'tommy\\tommy_all.csv', sep=';')

		for fold in range(self.k):
			cond = palmer_data.index.str.contains(self.palmer_folders[fold])
			palmer_fold = palmer_data.loc[cond]
			palmer_fold.to_csv(self.dest_folder + 'palmer\\' + self.palmer_folders[fold] + '.csv', sep=';')

			cond = tommy_data.index.str.contains(self.tommy_folders[fold])
			tommy_fold = tommy_data.loc[cond]
			tommy_fold.to_csv(self.dest_folder + 'tommy\\' + self.tommy_folders[fold] + '.csv', sep=';')

		self.organize_folds()
