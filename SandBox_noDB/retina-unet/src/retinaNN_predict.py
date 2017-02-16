###################################################
#
#   Script to
#   - Calculate prediction of the test dataset
#   - Calculate the parameters to evaluate the prediction
#
##################################################

#Python
import numpy as np
import configparser
from matplotlib import pyplot as plt
import os 
#os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=cpu,floatX=float32,optimizer=fast_compile'
#Keras
from keras.models import model_from_json
from keras.models import Model
#scikit learn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import sys
sys.path.insert(0, './lib/')
# help_functions.py
from help_functions import *
# extract_patches.py
from extract_patches import recompone
from extract_patches import recompone_overlap
from extract_patches import paint_border
from extract_patches import kill_border
from extract_patches import pred_only_FOV
from extract_patches import get_data_testing
from extract_patches import get_data_testing_overlap
# pre_processing.py
from pre_processing import my_PreProc


from databaseProxy import DatabaseProxy
#========= CONFIG FILE TO READ FROM =======
config = configparser.RawConfigParser()
config.read('configuration.txt')
#===========================================
#run the training on invariant or local
path_data = config.get('data paths', 'path_local')

#original test images (for FOV selection)
#DRIVE_test_imgs_original = path_data + config.get('data paths', 'test_imgs_original')
#test_imgs_orig = load_hdf5(DRIVE_test_imgs_original)
#full_img_height = test_imgs_orig.shape[2]
#full_img_width = test_imgs_orig.shape[3]
##the border masks provided by the DRIVE
#DRIVE_test_border_masks = path_data + config.get('data paths', 'test_border_masks')
#test_border_masks = load_hdf5(DRIVE_test_border_masks)
## dimension of the patches
#patch_height = int(config.get('data attributes', 'patch_height'))
#patch_width = int(config.get('data attributes', 'patch_width'))
##the stride in case output with average
#stride_height = int(config.get('testing settings', 'stride_height'))
#stride_width = int(config.get('testing settings', 'stride_width'))
#assert (stride_height < patch_height and stride_width < patch_width)
#model name
name_experiment = config.get('experiment name', 'name')
path_experiment = './' +name_experiment +'/'
#N full images to be predicted
Imgs_to_test = int(config.get('testing settings', 'full_images_to_test'))
#Grouping of the predicted images
N_visual = int(config.get('testing settings', 'N_group_visual'))
#====== average mode ===========
average_mode = config.getboolean('testing settings', 'average_mode')


# #ground truth
# gtruth= path_data + config.get('data paths', 'test_groundTruth')
# img_truth= load_hdf5(gtruth)
# visualize(group_images(test_imgs_orig[0:20,:,:,:],5),'original')#.show()
# visualize(group_images(test_border_masks[0:20,:,:,:],5),'borders')#.show()
# visualize(group_images(img_truth[0:20,:,:,:],5),'gtruth')#.show()



#============ Load the data and divide in patches
patches_imgs_test = None
new_height = None
new_width = None
masks_test  = None
patches_masks_test = None
#if average_mode == True:
#    patches_imgs_test, new_height, new_width, masks_test = get_data_testing_overlap(
#        DRIVE_test_imgs_original = DRIVE_test_imgs_original,  #original
#        DRIVE_test_groudTruth = path_data + config.get('data paths', 'test_groundTruth'),  #masks
#        Imgs_to_test = int(config.get('testing settings', 'full_images_to_test')),
#        patch_height = patch_height,
#        patch_width = patch_width,
#        stride_height = stride_height,
#        stride_width = stride_width
#    )
#else:
#    patches_imgs_test, patches_masks_test = get_data_testing(
#        DRIVE_test_imgs_original = DRIVE_test_imgs_original,  #original
#        DRIVE_test_groudTruth = path_data + config.get('data paths', 'test_groundTruth'),  #masks
#        Imgs_to_test = int(config.get('testing settings', 'full_images_to_test')),
#        patch_height = patch_height,
#        patch_width = patch_width,
#    )


db = DatabaseProxy()

#============ Load the data and divided in patches
_, _ , patches_imgs_test, patches_masks_test = db.getTestAndTrainingData(batches=32)


#================ Run the prediction of the patches ==================================
best_last = config.get('testing settings', 'best_last')
#Load the saved model
model = model_from_json(open(path_experiment+name_experiment +'_architecture.json').read())
model.load_weights(path_experiment+name_experiment + '_'+best_last+'_weights.h5')
#Calculate the predictions
predictions = model.predict(patches_imgs_test, batch_size=40, verbose=2)
print("predicted images size :")
print(predictions.shape)

#===== Convert the prediction arrays in corresponding images
pred_patches = pred_to_imgs(predictions,"original")


#========== Elaborate and visualize the predicted images ====================
#pred_imgs = None
#orig_imgs = None
#gtruth_masks = None
#if average_mode == True:
#    pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)# predictions
#    orig_imgs = my_PreProc(test_imgs_orig[0:pred_imgs.shape[0],:,:,:])    #originals
#    gtruth_masks = masks_test  #ground truth masks
#else:
#    print(pred_patches.shape)
#    pred_imgs = recompone(pred_patches,384,288)       # predictions
#    orig_imgs = recompone(patches_imgs_test,384,288)  # originals
#    gtruth_masks = recompone(patches_masks_test,384,288)  #masks
## apply the DRIVE masks on the repdictions #set everything outside the FOV to zero!!
#kill_border(pred_imgs, test_border_masks)  #DRIVE MASK  #only for visualization
### back to original dimensions
#orig_imgs = orig_imgs[:,:,0:full_img_height,0:full_img_width]
#pred_imgs = pred_imgs[:,:,0:full_img_height,0:full_img_width]
#gtruth_masks = gtruth_masks[:,:,0:full_img_height,0:full_img_width]
#print("Orig imgs shape: " +str(orig_imgs.shape))
#print("pred imgs shape: " +str(pred_imgs.shape))
#print("Gtruth imgs shape: " +str(gtruth_masks.shape))
#visualize(group_images(orig_imgs,N_visual),path_experiment+"all_originals")#.show()
#visualize(group_images(pred_imgs,N_visual),path_experiment+"all_predictions")#.show()
#visualize(group_images(gtruth_masks,N_visual),path_experiment+"all_groundTruths")#.show()
##visualize results comparing mask and prediction:
#assert (orig_imgs.shape[0]==pred_imgs.shape[0] and orig_imgs.shape[0]==gtruth_masks.shape[0])
#N_predicted = orig_imgs.shape[0]
#group = N_visual
#assert (N_predicted%group==0)
#for i in range(int(N_predicted/group)):
#    orig_stripe = group_images(orig_imgs[i*group:(i*group)+group,:,:,:],group)
#    masks_stripe = group_images(gtruth_masks[i*group:(i*group)+group,:,:,:],group)
#    pred_stripe = group_images(pred_imgs[i*group:(i*group)+group,:,:,:],group)
#    total_img = np.concatenate((orig_stripe,masks_stripe,pred_stripe),axis=0)
#    visualize(total_img,path_experiment+name_experiment +"_Original_GroundTruth_Prediction"+str(i))#.show()


#====== Evaluate the results
print("\n\n========  Evaluate the results =======================")
#predictions only inside the FOV
y_scores, y_true = pred_patches, patches_masks_test#pred_only_FOV(pred_imgs,gtruth_masks, test_border_masks)  #returns data only inside the FOV





#Confusion matrix
y_pred = y_scores





confusion = confusion_matrix(y_true.flatten(), y_pred.flatten())
print(confusion)

classification = classification_report(y_true.flatten(),y_pred.flatten())
print(classification)

accuracy = 0
if float(np.sum(confusion))!=0:
    accuracy = float(sum(np.diagonal(confusion)))/float(np.sum(confusion))
print("Global Accuracy: " +str(accuracy))


#Save the results
file_perf = open(path_experiment+'performances.txt', 'w')
file_perf.write("Confusion matrix:\n"
                +str(confusion)
                +"\nCLASSIFICATION_REPORT: "
                +str(classification)
                +"\nACCURACY: " +str(accuracy)
                )
file_perf.close()
