import numpy as np
import cv2

class MNGFeaturesFractal():

	def box_counting_dimension(self, gray_img):

		def get_box_sizes(gray_img):
			min_dim = min(gray_img.shape)
			n 		= int(np.floor(np.log(min_dim)/np.log(2)))
			sizes 	= 2**np.arange(n, 1, -1)
			return sizes

		def box_count(gray_img, k):
			img_sum = np.add.reduceat(np.add.reduceat(gray_img, np.arange(0, gray_img.shape[0], k), axis=0), np.arange(0, gray_img.shape[1], k), axis=1)

			return len(np.where((img_sum > 0) & (img_sum < k*k*255))[0])

		gray_img[gray_img != 255] = 0

		sizes = get_box_sizes(gray_img)
		
		counts 	= []
		for size in sizes:
			counts.append(box_count(gray_img, size))

		coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
		return -coeffs[0]

	def correlation_dimension(self, gray_img):

		def heaviside_func(array, threshold):
			return sum(array-threshold>=0)

		data 				= gray_img.flatten()
		n_pixels 			= data.shape[0]
		data_shifted 		= data[1:]
		data 				= data[0:n_pixels-1]
		data_subtracted		= data - data_shifted

		dists = np.linspace(0.1, 1, 5)

		C = []
		for dist in dists:
			sum_pixel = heaviside_func(data_subtracted, dist)
			C.append(sum_pixel/(n_pixels*n_pixels))

		coeffs = np.polyfit(np.log(C), np.log(dists), 1)
		return -coeffs[0]

	def dilation_dimension(self, gray_img):
		
		def invert_colors(gray_img):
			gray_img[gray_img==255] = 0
			gray_img[gray_img!=0] = 255
			return gray_img

		def get_radius_array(gray_img, h, w):
			min_dim = min(h, w)
			n = int(np.floor(np.log(min_dim)/np.log(2)))

			radius = np.arange(n, 0, -1)
			radius = radius*radius
			return radius

		def fill_image(gray_img, border, h, w):
			new_img = np.zeros((h+border*2,w+border*2), dtype='uint8')
			new_img[border:border+h,border:border+w] = gray_img
			return new_img

		def compute_sums(radius, k_size, gray_img):
			sums = []
			for r in radius:
				k = np.full((k_size,k_size), 0, dtype='uint8')
				new_k = cv2.circle(k,(int(k_size/2),int(k_size/2)), r, 1, -1)
				res = cv2.filter2D(gray_img, -1, new_k)
				sums.append(sum(sum(res)))

		h, w 	= gray_img.shape
		img 	= invert_colors(gray_img)

		radius 			= get_radius_array(gray_img, h, w)
		diameters 		= radius*2
		max_diameter 	= max(diameters)
		
		k_size 		= max_diameter-max_diameter%100+100
		border 		= max_diameter+50
		gray_img 	= fill_image(gray_img, border, h, w)

		sums = compute_sums(radius, k_size, gray_img)

		L = [s/d for s, d in zip(sums, diameters)]

		coeffs = np.polyfit(np.log(L), np.log(1/diameters), 1)
		return -coeffs[0]