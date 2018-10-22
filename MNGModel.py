from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

import pandas as pd
import math
import os

class MNGModel():

	def __init__(self, folder, folds):
		self.dest_folder 	= folder + '..results\\'
		self.folds 			= folds

	# def save_results(self, file, r2, rmse, proc_folder):

	def split_data(self, data):

		# X_train 	= data[1:n_cols - 1]
		# Y_train 	= data['target']

		# return X_train, Y_train

	def build_rf_model(self, proc_folder, n_trees=100):

		r2_values	= list()
		rmse_values	= list()

		for i in range(self.folds.k):
			train, test = self.folds.get_fold_data(i+1)

			X_train, Y_train 	= self.split_data(train)
			X_test, Y_test 		= self.split_data(test)

			rf_model 		= RandomForestClassifier(n_estimators=n_trees).fit(X_train, Y_train)
			Y_predicted 	= rf_model.predict(X_test)
			
			# rf_model.estimators_
			# rf_model.feature_importances_

			r2 		= r2_score(Y_test, Y_predicted)
			rmse 	= math.sqrt(mean_squared_error(Y_test, Y_predicted))
			
			r2_values.append(r2)
			rmse_values.append(rmse)

		save_results(file, r2_values, rmse_values, proc_folder)

	def build_mlr_model(self, proc_folder):
		
		r2_values	= list()
		rmse_values	= list()

		for i in range(self.folds.k):
			train, test = self.folds.get_fold_data(i+1)

			X_train, Y_train 	= self.split_data(train)
			X_test, Y_test 		= self.split_data(test)

			rf_model 		= LinearRegression().fit(X_train, Y_train)
			Y_predicted 	= rf_model.predict(X_test)
			
			r2 		= r2_score(Y_test, Y_predicted)
			rmse 	= math.sqrt(mean_squared_error(Y_test, Y_predicted))
			
			r2_values.append(r2)
			rmse_values.append(rmse)

		save_results(file, r2_values, rmse_values, proc_folder)
