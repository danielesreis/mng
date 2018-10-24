from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

import pandas as pd
import math
import os

class MNGModel():

	def __init__(self, folder, folds, model_type, att):
		self.dest_folder 	= folder + '..results\\'
		self.folds 			= folds
		self.model_type		= model_type
		self.att 			= att

	def save_results(self, algorithm, model_type, r2, rmse):

		results = pd.DataFrame({'r2': rmse, 'rmse': rmse}, index=np.arange(1,self.k+1))
		results.to_csv(self.dest_folder + algorithm + '_' + model_type + '.csv', sep=';')


	def split_data(self, data, features=['all']):

		if 'all' in features:
			X = data.drop(self.att, axis=1)
		else:
			X = data[features]

		Y 	= data[self.att]

		return X, Y

	def build_rf_model(self, model_type, n_trees=100):

		r2_values	= list()
		rmse_values	= list()

		for fold in range(self.folds.k):
			train, test = self.folds.get_fold_data(fold+1, self.model_type)

			X_train, Y_train 	= self.split_data(train)
			X_test, Y_test 		= self.split_data(test)

			rf_model 		= RandomForestClassifier(n_estimators=n_trees).fit(X_train, Y_train)
			Y_predicted 	= rf_model.predict(X_test)
			
			print(rf_model.estimators_)
			print(rf_model.feature_importances_)

			r2 		= r2_score(Y_test, Y_predicted)
			rmse 	= math.sqrt(mean_squared_error(Y_test, Y_predicted))
			
			r2_values.append(r2)
			rmse_values.append(rmse)

		save_results('rf', model_type, r2_values, rmse_values)

	def build_mlr_model(self, model_type, features):
		
		r2_values	= list()
		rmse_values	= list()

		for fold in range(self.folds.k):
			train, test = self.folds.get_fold_data(fold+1)

			X_train, Y_train 	= self.split_data(train, features)
			X_test, Y_test 		= self.split_data(test)

			rf_model 		= LinearRegression().fit(X_train, Y_train)
			Y_predicted 	= rf_model.predict(X_test)
			
			r2 		= r2_score(Y_test, Y_predicted)
			rmse 	= math.sqrt(mean_squared_error(Y_test, Y_predicted))
			
			r2_values.append(r2)
			rmse_values.append(rmse)

		save_results('mlr', model_type, r2_values, rmse_values)
