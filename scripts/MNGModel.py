from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import scipy
import math
import os

class MNGModel():

	def __init__(self, folder, file_path, model_type, att):
		self.dest_folder 	= folder + '..\\results\\'
		self.data			= pd.read_csv(file_path, sep=';', index_col=0)
		self.model_type		= model_type
		self.att 			= att

	def save_results(self, algorithm, model_type, r2, rmse):

		results = pd.DataFrame(columns=['r2', 'rmse'])
		results['r2'] = [r2]
		results['rmse'] = [rmse]
		# results = pd.DataFrame({'r2': r2, 'rmse': rmse}, index=np.arange(1,self.folds.k+1))
		# results.index.name = 'fold'
		results.to_csv(self.dest_folder + algorithm + '_' + model_type + '.csv', sep=';')

	# def split_data(self, data, features=['all']):

	# 	if 'all' in features:
	# 		X = data.drop(self.att, axis=1)
	# 	else:
	# 		X = data[features]

	# 	Y 	= data[self.att]

	# 	return X, Y

	def build_rf_model(self, model_type, n_trees=100):

		# r2_values	= list()
		# rmse_values	= list()

		X = self.data.drop(self.att, axis=1)
		Y = self.data[self.att]

		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

		rf_model 		= RandomForestRegressor(n_estimators=n_trees).fit(X_train, Y_train)
		Y_predicted 	= rf_model.predict(X_test)
			
		# print(rf_model.estimators_)
		r2 	= r2_score(Y_test, Y_predicted)
		rmse= math.sqrt(mean_squared_error(Y_test, Y_predicted))

		self.save_results('rf_'+str(n_trees), model_type, r2, rmse)

		# for fold in range(self.folds.k):
		# 	train, test = self.folds.get_fold_data(fold+1, self.model_type)

		# 	X_train, Y_train 	= self.split_data(train)
		# 	X_test, Y_test 		= self.split_data(test)

			# rf_model 		= RandomForestRegressor(n_estimators=n_trees).fit(X_train, Y_train)
			# Y_predicted 	= rf_model.predict(X_test)
			
			# # print(rf_model.estimators_)
			# print(rf_model.feature_importances_)

		# 	r2 		= r2_score(Y_test, Y_predicted)
		# 	rmse 	= math.sqrt(mean_squared_error(Y_test, Y_predicted))
			
		# 	r2_values.append(r2)
		# 	rmse_values.append(rmse)

		# self.save_results('rf', model_type, r2_values, rmse_values)

	def build_mlr_model(self, model_type, features):
		
		# r2_values	= list()
		# rmse_values	= list()

		new_data = self.data.copy()

		X = new_data.drop(self.att, axis=1)
		Y = new_data[self.att]

		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

		mlr_model 		= LinearRegression().fit(X_train, Y_train)
		Y_predicted 	= mlr_model.predict(X_test)
			
		r2 	= r2_score(Y_test, Y_predicted)
		rmse= math.sqrt(mean_squared_error(Y_test, Y_predicted))

		self.save_results('mlr', model_type, r2, rmse)

		# for fold in range(self.folds.k):
		# 	train, test = self.folds.get_fold_data(fold+1, self.model_type)

		# 	X_train, Y_train 	= self.split_data(train, features)
		# 	X_test, Y_test 		= self.split_data(test, features)

		# 	mlr_model 		= LinearRegression().fit(X_train, Y_train)
		# 	Y_predicted 	= mlr_model.predict(X_test)
			
		# 	r2 		= r2_score(Y_test, Y_predicted)
		# 	rmse 	= math.sqrt(mean_squared_error(Y_test, Y_predicted))
			
		# 	r2_values.append(r2)
		# 	rmse_values.append(rmse)

		# self.save_results('mlr', model_type, r2_values, rmse_values)
