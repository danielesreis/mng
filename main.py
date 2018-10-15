import MNG_PreProcessing
import MNG_Segmentation
import MNG_Contour
import MNG_Features
import MNG_Model
import MNG

import os
import cv2

path 			= os.getcwd()
folder 			= path + '\\images\\'
img_names 		= os.listdir(folder)
proc_folders 	= ['original\\', 'median\\', 'deblurring\\', 'opening\\', 'closing\\', 'sharpening\\']

MNG 		= MNG(path)

# get folder and index
for proc_folder in proc_folders:

	MNG.features.delete_rows()
	func = get_processing_func(proc_folder)

	for img_name in img_names:
		BGR_img = cv2.imread(img_name)
		# # compare to nonetype
		# if type(func) is not:
		# 	filt_img = func(BGR_img)
		# else:
		# 	filt_img = BGR_img 

		filt_img 	= MNG.segmentation.otsu_thresholding(filt_img)
		filt_img 	= MNG.preprocessing.remove_shadow(filt_img)
		filt_img 	= MNG.contour.cut_image(ori_img)
		seg_img 	= MNG.preprocessing.rotate(filt_img)

		MNG.save_image(img_name, filt_img, path + '\\preprocessing\\' + proc_folder)
		MNG.save_image(img_name, seg_img, path + '\\segmentation\\' + proc_folder)

		MNG.features.extract_features(seg_img, img_name)

	features = MNG.features.save_data(proc_folder)
	MNG.model.build_rf_model(features)
	MNG.model.build_mlr_model(features)

# def get_processing_func(preprocessing_name):
# 	if preprocessing_name if 'original\\' :
# 		return 
# 	elif preprocessing_name if 'median\\':
# 		return MNG.preprocessing.median_filter
# 	elif preprocessing_name if 'deblurring\\':
# 		return MNG.preprocessing.deblurring
# 	elif preprocessing_name if 'opening\\':
# 		return MNG.preprocessing.opening_operation
# 	elif preprocessing_name if 'closing\\':
# 		return MNG.preprocessing.closing_operation
# 	elif preprocessing_name if 'sharpening\\':
# 		return MNG.preprocessing.unsharp_masking





