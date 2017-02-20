#########################################################
# Given a a test image will return classified image
# If given image is part of dataset will also return truth data
#
#########################################################

import sys, getopt, os, configparser

from PIL import Image

import numpy as np

from extract_patches import get_data_testing

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
    truth = check_dataset(img)

    # Convert img to correct format to be handled
    img = np.asarray(img)
    img = np.asarray([img])
    img = np.transpose(img, (0,3,1,2))

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
    pred_img = pred_to_imgs(predictions,mode="original", N_classes=N_classes,N_images=1)

    # Tranform back into image
    classified = np.transpose(pred_img, (0,3,2,1))
    classified = classified[0]
    classified = np.reshape(classified,(classified.shape[0],classified.shape[1]))
    classified = Image.fromarray((classified*255).astype(np.uint8))

    return classified, truth

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

    classified, truth = classify_image(img,model)

    # Display images
    img.show()
    if classified:
        classified.show()
    if truth:
        truth.show()

    return 0

if __name__ == '__main__':
    main()
