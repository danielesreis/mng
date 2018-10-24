import numpy as np
import pandas as pd
import os

class MNGFolds():

	def __init__(self, folder, features_path, k):
		self.origin_folder	 	= folder
		self.dest_folder	 	= folder + '..\\features\\'
		self.features_path		= features_path
		self.k	 				= k
		self.data	 			= self.set_data(features_path)
		self.n	 				= len(self.data)
		self.palmer_folders,\
		self.tommy_folders 		= self.set_folders_names(k)

	def set_data(self, features_path):
		return pd.read_csv(features_path, sep=';', index_col=0)

	def set_folders_names(self, k):

		palmer_names 	= ['palmer']*self.k
		tommy_names 	= ['tommy']*self.k

		nums = list(np.arange(1, self.k+1))

		palmer_folders 	= [palmer+str(num) for palmer,num in zip(palmer_names,nums)]
		tommy_folders 	= [tommy+str(num) for tommy,num in zip(tommy_names,nums)]

		return palmer_folders, tommy_folders

	def get_fold_data(self, fold, model_type):

		if model_type is 'all':
			train 	= pd.read_csv(self.dest_folder + 'all\\train_sem_' + str(fold))
			test 	= pd.read_csv(self.dest_folder + 'all\\test_' + str(fold))
		elif model_type is 'palmer':
			train 	= pd.read_csv(self.dest_folder + 'palmer\\train_sem_' + str(fold))
			test 	= pd.read_csv(self.dest_folder + 'palmer\\test_' + str(fold))
		else:
			train 	= pd.read_csv(self.dest_folder + 'tommy\\train_sem_' + str(fold))
			test 	= pd.read_csv(self.dest_folder + 'tommy\\test_' + str(fold))
				
		return train, test

	def organize_folds(self):
		nums = np.arange(1, self.k+1)

		for fold in range(self.k):

			palmer_test 	= pd.read_csv(self.dest_folder + 'palmer\\' + palmer_folders[fold])
			tommy_test 		= pd.read_csv(self.dest_folder + 'tommy\\' + tommy_folders[fold])
			all_test		= pd.concat([palmer,tommy])
			all_test.to_csv(self.dest_folder + 'all\\test_' + str(fold+1))

			train_folds = np.setdiff1d(nums, fold+1)

			palmer_train 	= pd.DataFrame([])
			tommy_train 	= pd.DataFrame([])
			all_train 		= pd.DataFrame([])
			for train_fold in train_folds:
				palmer_fold_train 	= pd.read_csv(self.dest_folder + 'palmer\\' + palmer_folders[train_fold])
				tommy_fold_train 	= pd.read_csv(self.dest_folder + 'tommy\\' + tommy_folders[train_fold])

				palmer_train 	= pd.concat([palmer_train, palmer_fold_train])
				tommy_train 	= pd.concat([tommy_train, tommy_fold_train])
				all_train 		= pd.concat([all_train, palmer_fold_train, tommy_fold_train])

			palmer_train.to_csv(self.dest_folder + 'palmer\\train_sem_' + str(fold+1))
			palmer_train.to_csv(self.dest_folder + 'palmer\\train_sem_' + str(fold+1))
			all_train.to_csv(self.dest_folder + 'all\\train_sem_' + str(fold+1))

	def separate_folds(self):

		palmer_cond = self.data.index.str.contains('palmer')
		tommy_cond 	= self.data.index.str.contains('tommy')

		palmer_data = self.data.loc[palmer_cond]
		tommy_data 	= self.data.loc[tommy_cond]

		palmer_data.to_csv(self.dest_folder + 'palmer\\palmer_all.csv', sep=';')
		tommy_data.to_csv(self.dest_folder + 'tommy\\tommy_all.csv', sep=';')

		n_palmer 	= int(len(palmer_data)/self.k)
		n_tommy 	= int(len(tommy_data)/self.k)

		for fold in range(self.k):
			cond 		= self.data.index.str.contains(palmer_folders[fold])
			palmer_fold = palmer_data.loc[cond]
			palmer_fold.to_csv(self.dest_folder + 'palmer\\' + self.palmer_folders[fold] + '.csv', sep=';')

			cond 		= self.data.index.str.contains(tommy_folders[fold])
			tommy_fold = tommy_data.loc[cond]
			tommy_fold.to_csv(self.dest_folder + 'tommy\\' + self.tommy_folders[fold] + '.csv', sep=';')
