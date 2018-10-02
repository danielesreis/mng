from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import pandas as pd
import math
import os

path 			= os.getcwd() + '\\feature_extraction\\'
files			= os.listdir(path)
dest_folder		= path + '\\..\\'

for file in files:
	# check this
	data 		= pd.read_csv(path + file, sep = ';')
	n_cols 		= len(data.columns)
	X_train 	= data[1:n_cols - 1]
	Y_train 	= data['target']

	# Random Forest
	rf_model 	= RandomForestClassifier(n_estimators = 100)
	# rf_model.estimators_
	# rf_model.feature_importances_

	r2_score 	= cross_val_score(rf_model, X_train, Y_train, cv = 6, scoring = 'r2')
	mse_score 	= cross_val_score(rf_model, X_train, Y_train, cv = 6, scoring = 'mean_squared_error')
	rmse_score 	= math.sqrt(mse_score)
	# r2_score.mean()
	# rmse_score.mean()

	# Linear Regression
	for i in range(n_cols - 2):

		X_feature 	= X_train[i]
		reg 		= LinearRegression()

		r2_score 	= cross_val_score(reg, X_train, Y_train, cv = 6, scoring = 'r2')
		mse_score 	= cross_val_score(reg, X_train, Y_train, cv = 6, scoring = 'mean_squared_error')
		rmse_score 	= math.sqrt(mse_score)
		# r2_score.mean()
		# rmse_score.mean()
