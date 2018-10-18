class MNGFeaturesDominantColor():
	
	def __init__(self, features_means):
		self.features_means = features_means

	def dominant_HSV_color(self, img):
	''' 0: red, 1: yellow, 2: green, 3: cyan, 4: blue, 5: magenta '''
	means 	= self.features_means.channels_mean(img)	
	hue 	= means[0]

	if 15 <= hue < 45:
		dominant_color = 1
	elif 45 <= hue < 75:
		dominant_color = 2
	elif 75 <= hue < 105:
		dominant_color = 3
	elif 105 <= hue < 135:
		dominant_color = 4
	elif 135 <= hue < 165:
		dominant_color = 5
	else: 
		dominant_color = 0

	return dominant_color