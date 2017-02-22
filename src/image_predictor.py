#########################################################
# Given a a test image will return classified image
# If given image is part of dataset will also return truth data
#
#########################################################

import sys, getopt, os, configparser, ipdb

from PIL import Image

import numpy as np

from extract_patches import *

from keras.models import model_from_json
from keras.models import Model

from pre_processing import my_PreProc

from help_functions import *

DATA_DIR = '../Dataset/FinalRawData/'
LABELED_DIR = '../Dataset/FinalLabeledData/'

config = configparser.RawConfigParser()
config.read('configuration.txt')
name_experiment = config.get('experiment name', 'name')
path_experiment = './' +name_experiment +'/'
best_last = config.get('testing settings', 'best_last')

N_classes = int(config.get('training settings', 'N_classes'))

def classify_image(img,model):
    ''' Classifies image given a model '''
    # Check img to see if truth data is avaliable
    original = check_dataset(img)

    # Convert img to correct format to be handled
    img = np.asarray(img)
    img = np.asarray([img])
    img = np.transpose(img, (0,3,1,2))

    #true = np.asarray(truth)
    #true = np.asarray([true])
    #ipdb.set_trace()
    #true = np.transpose(true, (0,3,2,1))

    # Split into patches
    patches_imgs_test, patches_masks_test = get_data_testing(
        p_test_imgs_original = img,
        p_test_groudTruth = img,  #masks SET TO img BECAUSE DONT MATTER
        Imgs_to_test = 1,
        patch_height = int(config.get('data attributes', 'patch_height')),
        patch_width = int(config.get('data attributes', 'patch_width')),
    )


    # Calculate Prediction
    predictions = model.predict(patches_imgs_test, batch_size=40, verbose=2)
    qq = predictions.reshape(108,  32, 32, 4)
    q = np.zeros((108,32,32,1))

    for i in range(qq.shape[0]):
        for j in range(qq.shape[1]):
            for k in range(qq.shape[1]):
                q[i,j,k] = np.argmax(qq[i,j,k])
    q = q.transpose((0,3,1,2))
    q = recompone(q,1,1)
    q = q / N_classes
    q = q * 255
    pred_img = q
    #pred_img = pred_to_imgs(predictions,mode="original", N_classes=N_classes,N_images=1)



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
    

    # Tranform back into image
    #pred_img = recompone(pred_img, 1, 1)
    #patches_masks_test = recompone(patches_masks_test, 1, 1)
    #truth = np.transpose(patches_masks_test, (0,3,2,1))
    #truth = truth[0]
    #truth = truth / N_classes
    #truth = Image.fromarray((truth*255).astype(np.uint8))

    #classified = np.transpose(pred_img, (0,2,3,1))
    #classified = classified[0].repeat(3,2)
    #classified = classified / N_classes
    classified = group_images(pred_img,9)
    classified = classified.transpose((1,0,2))
    classified = visualize(classified,"classifiedImgPatches.png")
    truth = group_images(patches_imgs_test,9)
    truth = truth.transpose((1,0,2))
    truth = visualize(truth,"inputImgPatches.png")


    return classified, truth, original

def check_dataset(img):
    ''' Checks dataset to see if imagine truth data is avalible '''
    abs_path = os.path.dirname(os.path.abspath(__file__))
    path_to_labeled = os.path.join(abs_path, LABELED_DIR)
    path_to_raw = os.path.join(abs_path, DATA_DIR)
    
    truth_images = sorted(os.listdir(path_to_labeled))
    raw_images = sorted(os.listdir(path_to_raw))

    for raw, labeled in zip(raw_images,truth_images):
        check_img = open_image(path_to_raw+raw)
        if check_img == img:
            truth_img = open_image(path_to_labeled+labeled)
            print("Found match in truth data for given image")
            return truth_img

    return

def open_image(path):
    ''' Opens image file as PIL '''
    try:
        return Image.open(path)
    except:
        print("Unable to open image: \n{0} \nExiting!".format(path))
        exit(0)

def handle_arguments():
    ''' Handles command line arguments '''
    if len(sys.argv) < 3:
        print_usage()

    myopts, args = getopt.getopt(sys.argv[1:],"i:h:")
    img_path=''

    for o, a in myopts:
        if o == '-i':
            img_path=a
        elif o == '-h':
            print_usage()
        else:
            print_usage()

    print("\nImage path: {0}\n".format(img_path))
    img = open_image(img_path) 

    return img
            
def print_usage():
    print("Usage: {0} -i <input image>".format(sys.argv[0]))
    exit(0)


def main():
    img = handle_arguments()

    # Load model
    model = model_from_json(open(path_experiment+name_experiment +'_architecture.json').read())
    model.load_weights(path_experiment+name_experiment + '_'+best_last+'_weights.h5')

    classified, truth, original= classify_image(img,model)

    # Display images
    img.show()
    if classified:
        classified.show()
    if truth:
        truth.show()
    if original:
        original.show()

    return 0

if __name__ == '__main__':
    main()
