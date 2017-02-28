#########################################################
# Given a a test image will return classified image
# If given image is part of dataset will also return truth data
#
#########################################################



#DO YOU THINK THAT IT WILL BE EASY TO MAKE THIS INTO A CLASS ISNTEAD OF JUST A  FILE

import sys, getopt, os, configparser, ipdb

from PIL import Image, ImageOps

import numpy as np

from extract_patches import *

from keras.models import model_from_json
from keras.models import Model

from pre_processing import my_PreProc

from help_functions import *


def open_image(path):
    ''' Opens image file as PIL '''
    try:
        return Image.open(path)
    except:
        print("Unable to open image: \n{0} \nExiting!".format(path))
        exit(0)


class ImagePredictor():
    def __init__(self, configFile="configuration.txt", data_dir='../Dataset/FinalRawData/', labeled_dir='../Dataset/FinalLabeledData/'):
        config = configparser.RawConfigParser()
        config.read(configFile)
        self.name_experiment = config.get('experiment name', 'name')
        self.path_experiment = './' +self.name_experiment +'/'
        self.best_last = config.get('testing settings', 'best_last')
        self.N_classes = int(config.get('training settings', 'N_classes'))
        self.patch_height = int(config.get('data attributes', 'patch_height')),
        self.patch_width = int(config.get('data attributes', 'patch_width')),
        self.model = model_from_json(open(self.path_experiment+self.name_experiment +'_architecture.json').read())
        self.model.load_weights(self.path_experiment+self.name_experiment + '_'+self.best_last+'_weights.h5')
        self.DATA_DIR = data_dir
        self.LABELED_DIR = labeled_dir

    def check_dataset(self,img):
        ''' Checks dataset to see if imagine truth data is avalible '''
        abs_path = os.path.dirname(os.path.abspath(__file__))
        path_to_labeled = os.path.join(abs_path, self.LABELED_DIR)
        path_to_raw = os.path.join(abs_path, self.DATA_DIR)
        
        truth_images = sorted(os.listdir(path_to_labeled))
        raw_images = sorted(os.listdir(path_to_raw))

        for raw, labeled in zip(raw_images,truth_images):
            check_img = open_image(path_to_raw+"/"+raw)
            if check_img == img:
                truth_img = open_image(path_to_labeled+"/"+labeled)
                print("Found match in truth data for given image")
                return truth_img

        return

    def classify_image(self, img):
        ''' Classifies image given a model '''
        # Check img to see if truth data is avaliable
        original = self.check_dataset(img)

        # Convert img to correct format to be handled
        img = np.asarray(img)
        img = np.asarray([img])
        img = np.transpose(img, (0,3,1,2))

        # Split into patches
        patches_imgs_test, patches_masks_test = get_data_testing(
            p_test_imgs_original = img,
            p_test_groudTruth = img,  #masks SET TO img BECAUSE DONT MATTER
            Imgs_to_test = 1,
            patch_height = self.patch_height[0],
            patch_width = self.patch_width[0]
        )

        # Calculate Prediction
        predictions = self.model.predict(patches_imgs_test, batch_size=40, verbose=2)
        qq = predictions.reshape(108,  32, 32, 4)
        q = np.zeros((108,32,32,1))

        for i in range(qq.shape[0]):
            for j in range(qq.shape[1]):
                for k in range(qq.shape[1]):
                    q[i,j,k] = np.argmax(qq[i,j,k])
        q = q.transpose((0,3,1,2))
        
        #q = recompone(q,1,1)
        q = q / self.N_classes
        q = q * 255
        pred_img = q

        classified = group_images(pred_img,12)
        classified = classified.transpose((1,0,2))
        classified = visualize(classified,self.path_experiment + "classifiedImgPatches",flippy=True)
        truth = group_images(patches_imgs_test,12)
        truth = truth.transpose((1,0,2))
        truth = visualize(truth, self.path_experiment+ "inputImgPatches", flippy=True)


        return classified, truth, original




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
    predictor = ImagePredictor(configFile="./configuration.txt", )

    img = handle_arguments()

    # Load model
    classified, truth, original= predictor.classify_image(img)

    # Display images
    img.show()
    if classified:
        classified.show()
        classified.save(predictor.path_experiment+ "classifiedImgPatches.png")
    if truth:
        truth.show()
        truth.save(predictor.path_experiment + "inputImgPatches.png")
    if original:
        original.show()

    return 0

if __name__ == '__main__':
    main()
