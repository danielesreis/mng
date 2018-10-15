import cv2

def MNG_Segmentation():

	def __init__(self, folder):
		self.dest_folder = folder + 'segmentation\\'	

	def simple_thresholding(color_img, threshold):
		grayscale_img 			= cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
		__, mask 				= cv2.threshold(grayscale_img, threshold, 255, cv2.THRESH_BINARY)
		inv_mask 				= cv2.bitwise_not(mask)
		thr_img 				= cv2.bitwise_and(color_img, color_img, mask = inv_mask)
		thr_img[thr_img == 0] 	= 255

		return thr_img

	def otsu_thresholding(color_img):
		grayscale_img 			= cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
		__, mask 				= cv2.threshold(grayscale_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		inv_mask				= cv2.bitwise_not(mask)
		thr_img 				= cv2.bitwise_and(color_img, color_img, mask = inv_mask)
		thr_img[thr_img == 0] 	= 255
	
		return thr_img