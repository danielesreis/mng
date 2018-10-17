import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

class MNGFeaturesGradient():

	def longitudinal_gradient(self, img, OFFSET_LONG=0.05, WIDTH_FRACTION=0.05):
		height, width, __ 	= img.shape

		def first_last_y(height):
			y_i	= round(OFFSET_LONG * height)
			y_f	= height - round(OFFSET_LONG * height)
			return y_i, y_f

		def first_last_x(width):
			x_c					= round(width / 2)
			slice_width			= round(width * WIDTH_FRACTION / 2)
			x_f					= x_c + slice_width
			x_i					= x_c - slice_width
			return x_i, x_f

		def get_X_Y(data):
			X 	= np.arange(1, data.shape[0]+1, 1).transpose()
			X 	= np.array([X]).transpose()
			Y 	= [data[:,i] for i in range(3)]
			return X, Y
				
		y_i, y_f 	= first_last_y(height)
		x_i, x_f 	= first_last_x(width)
		l_slice		= img[y_i:y_f,x_i:x_f,:]
		data 		= np.mean(l_slice, axis=1)
		X, Y 		= get_X_Y(data)
	
		REG 	= [LinearRegression().fit(X, y) for y in Y]
		PRED 	= [reg.predict(X) for reg in REG]

		MSE 	= list()
		for i in range(3):
		    MSE.append(mean_squared_error(Y[i], PRED[i]))
		    
		ind 		= MSE.index(min(MSE))
		gradient 	= REG[ind].coef_[0]

		return gradient