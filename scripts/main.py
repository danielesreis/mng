import os
import cv2
from MNG import MNG
from MNGFolds import MNGFolds
from MNGModel import MNGModel

k 				= 7
model_type 		= 'all'
att 			= 'sst'

palmer_sst 		= 'C:\\Users\\juju-\\Desktop\\projetos\\mng\\palmer_sst.csv'
tommy_sst 		= 'C:\\Users\\juju-\\Desktop\\projetos\\mng\\tommy_sst.csv'

path 			= os.getcwd()
folder 			= path + '\\..\\images\\'
img_names 		= os.listdir(folder)

MNG 			= MNG(path, img_names, att)

feature_names 	= MNG.features.feature_names
features_mlr 	= feature_names[:9]
MNG.features.current_features = features_mlr
MNG.features.current_features_name = 'means'
MNG.features.new_df()

def pre_process(img_name):
	BGR_img = cv2.imread(folder+img_name)

	med 	= MNG.preprocessing.median_filter(BGR_img, 11)
	seg, __ = MNG.segmentation.otsu_thresholding(med)
	ope 	= MNG.preprocessing.opening_operation(seg, np.ones((15,15), np.uint8))
	shd 	= MNG.preprocessing.remove_shadow(ope)
	filt 	= MNG.preprocessing.closing_operation(shd, med, np.ones((13,13), np.uint8))

	filt[filt==0] = 255
	cut = MNG.contour.cut_image(filt)
	return cut
	
for img_name in img_names:

	filt_img = pre_process(img_name)
	MNG.save_image(img_name, filt_img, path + '\\preprocessadas\\')
	MNG.features.extract_features(seg_img, img_name)

file_path = MNG.features.save_data()
MNG.features.add_target(file_path, palmer_sst, tommy_sst)
MNG.folds = MNGFolds(folder, file_path, k)
MNG.folds.separate_folds()

MNG.model = MNGModel(folder, MNG.folds, model_type, att)
MNG.model.build_rf_model(model_type)

MNG.model.build_mlr_model(model_type, features_mlr)