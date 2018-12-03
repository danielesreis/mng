import numpy as np
import cv2

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

class MNGFeaturesGradient():

	def longitudinal_gradient(self, img, OFFSET_LONG=0.05, WIDTH_FRACTION=0.05):

		def rotate(img, h, w):
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

			first_slice = gray[0,:]
			last_slice 	= gray[-1,:]

			i, 		= np.where(first_slice!=255)
			top_mid = int((i[-1]-i[0])/2)+i[0]

			i, 		= np.where(last_slice!=255)
			bot_mid = int((i[-1]-i[0])/2)+i[0]			

			ang = math.tan(abs(top_mid-bot_mid)/h)*180
			M 	= cv2.getRotationMatrix2D((w/2,h/2),ang,1)
			dst = cv2.warpAffine(img,M,(w,h))

			return dst

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
				
		height, width, __ 	= img.shape
		img 		= rotate(img, height, width)
		
		y_i, y_f 	= first_last_y(height)
		x_i, x_f 	= first_last_x(width)
		l_slice		= img[y_i:y_f,x_i:x_f,:]
		data 		= np.mean(l_slice, axis=1)
		X, Y 		= get_X_Y(data)

		return np.array([x_i,x_f,y_i,y_f])
	
		# REG 	= [LinearRegression().fit(X, y) for y in Y]
		# PRED 	= [reg.predict(X) for reg in REG]

		# MSE 	= list()
		# for i in range(3):
		#     MSE.append(mean_squared_error(Y[i], PRED[i]))
		    
		# ind 		= MSE.index(min(MSE))
		# gradient 	= REG[ind].coef_[0]

		# return gradient