import os
import cv2
import MNG

k 				= 7
model_type 		= 'all'

MNG 			= MNG(path, img_names)
MNG.features.new_df()
feature_names 	= MNG.features.feature_names
# means
features_mlr 	= feature_names[:8]

path 			= os.getcwd()
folder 			= path + '\\images\\'
img_names 		= os.listdir(folder)


func = get_processing_func(proc_folder)
for img_name in img_names:
	BGR_img = cv2.imread(img_name)

	# build model for original images too
	filt_img = func(BGR_img)

	seg_img = MNG.segmentation.otsu_thresholding(filt_img)
	seg_img = MNG.preprocessing.remove_shadow(filt_img)
	seg_img = MNG.contour.cut_image(seg_img)

	MNG.save_image(img_name, filt_img, path + '\\preprocessing\\')
	MNG.save_image(img_name, seg_img, path + '\\segmentation\\')

	MNG.features.extract_features(seg_img, img_name)

file_path = MNG.features.save_data()
MNG.folds = MNGFolds(folder, file_path, k)
MNG.folds.separate_folds()

MNG.model = MNGModel(folder, MNG.folds, model_type)
# get sst values
MNG.model.build_rf_model(model_type)
# divide features data frame into smaller dataframes and call build_mlr_model for each smaller dataframe

MNG.model.build_mlr_model(model_type, features, mlr)

def get_processing_func(preprocessing_name):
	if preprocessing_name == 'original\\':
		return -1
	elif preprocessing_name == 'median\\':
		return MNG.preprocessing.median_filter
	elif preprocessing_name == 'deblurring\\':
		return MNG.preprocessing.deblurring_filter
	elif preprocessing_name == 'opening\\':
		return MNG.preprocessing.opening_operation
	elif preprocessing_name == 'closing\\':
		return MNG.preprocessing.closing_operation
	elif preprocessing_name == 'sharpening\\':
		return MNG.preprocessing.unsharp_masking