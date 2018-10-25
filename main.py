import os
import cv2
from MNG import MNG
from MNGFolds import MNGFolds
from MNGModel import MNGModel

k 				= 7
model_type 		= 'all'
att 			= 'sst'

path 			= os.getcwd()
folder 			= path + '\\images\\'
img_names 		= os.listdir(folder)

MNG 			= MNG(path, img_names)

feature_names 	= MNG.features.feature_names
features_mlr 	= feature_names[:9]
MNG.features.current_features = features_mlr
MNG.features.new_df()

for img_name in img_names:
	BGR_img = cv2.imread(folder+img_name)

	# preprocess img

	seg_img = MNG.segmentation.otsu_thresholding(filt_img)
	seg_img = MNG.preprocessing.remove_shadow(filt_img)
	seg_img = MNG.contour.cut_image(seg_img)

	MNG.save_image(img_name, filt_img, path + '\\preprocessing\\')
	MNG.save_image(img_name, seg_img, path + '\\segmentation\\')

	MNG.features.extract_features(seg_img, img_name)

file_path = MNG.features.save_data()
MNG.folds = MNGFolds(folder, file_path, k)
MNG.folds.separate_folds()

MNG.model = MNGModel(folder, MNG.folds, model_type, att)
# get sst values
MNG.model.build_rf_model(model_type)
# divide features data frame into smaller dataframes and call build_mlr_model for each smaller dataframe

MNG.model.build_mlr_model(model_type, features_mlr)