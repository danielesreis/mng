from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_scor

import pandas as pd
import math
import os

class MNGModel():

	def __init__(self, folder):
		self.dest_folder = folder + '..results\\'

	# def save_results(self, file, r2, rmse, proc_folder):

	def read_data(self, file):
		data = pd.read_csv(file, sep=';')	
		X_train 	= data[1:n_cols - 1]
		Y_train 	= data['target']

		return X_train, Y_train

	def build_rf_model(self, file, proc_folder, n_trees=100):
		X_train, Y_train = read_data(file)
		n_cols 		= len(X_train.columns)

		rf_model 	= RandomForestClassifier(n_estimators=n_trees).fit(X_train, Y_train)
		# rf_model.estimators_
		# rf_model.feature_importances_

		# r2_score 	= cross_val_score(rf_model, , , cv = 6, scoring = 'r2')
		# mse_score 	= cross_val_score(rf_model, , , cv = 6, scoring = 'mean_squared_error')
		# rmse_score 	= math.sqrt(mse_score)
		# r2 = r2_score.mean()
		# rmse = rmse_score.mean()
		save_results(file, r2, rmse, proc_folder)

	def build_mlr_model(self, file, proc_folder):
		X_train, Y_train = read_data(file)
		n_cols 	= len(X_train.columns)

		reg	= LinearRegression().fit(X_feature, Y_train)

		# 	# r2_score 	= cross_val_score(reg, , , cv = 6, scoring = 'r2')
		# 	# mse_score 	= cross_val_score(reg, , , cv = 6, scoring = 'mean_squared_error')
		# 	# rmse_score 	= math.sqrt(mse_score)
		# 	# r2_score.mean()
		# 	# rmse_score.mean()
		save_results(file, r2, rmse, proc_folder)
