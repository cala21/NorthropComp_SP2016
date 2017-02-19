###################################################
#
#   Script to pre-process the original imgs
#
##################################################


import numpy as np
from PIL import Image, ImageEnhance
import cv2
import ipdb
from help_functions import *


# My pre processing (use for both training and testing!)
# AS OF 2/14/17 the best combo is PILContrast and PILSharpen
def my_PreProc(data, saveImage=False, experiment_name=None): 
    assert(len(data.shape) == 4)
    assert (data.shape[1] == 3)  # Use the original images
    # black-white conversion

    train_imgs = rgb2gray(data)
    image_stages = tuple()

    gray = np.transpose(train_imgs[0], (1, 2, 0))
    image_stages += (gray,)
    # my preprocessing:

    #train_imgs = clahe_equalized(train_imgs)
    #clahe = np.transpose(train_imgs[0], (1, 2, 0))
    #image_stages += (clahe,)

    #train_imgs = dataset_normalized(train_imgs)
    #normalized = np.transpose(train_imgs[0], (1, 2, 0))
    #image_stages += (normalized,)

    #train_imgs = adjust_gamma(train_imgs, 1.2)
    #gamma = np.transpose(train_imgs[0], (1, 2, 0))
    #image_stages += (gamma,)

    #train_imgs = gaussian_edge_sharpening(train_imgs)
    #sharpen = np.transpose(train_imgs[0], (1, 2, 0))
    #image_stages += (sharpen,)

    train_imgs = PILContrast(train_imgs, 2)
    contrast = np.transpose(train_imgs[0], (1, 2, 0))
    image_stages += (contrast,)

    train_imgs = PILSharpness(train_imgs, 2)
    sharpness = np.transpose(train_imgs[0], (1, 2, 0))
    image_stages += (sharpness,)

#    train_imgs = gray2rgb(train_imgs)
#    final = np.transpose(train_imgs[0], (1, 2, 0))

    if(saveImage and experiment_name):
        all = np.hstack(image_stages).repeat(3,2)
        Image.fromarray(all.astype(np.uint8)).save(
            "./" + experiment_name + "/" + experiment_name + "_final_preprocessing.png")

    return train_imgs


#============================================================
#========= PRE PROCESSING FUNCTIONS ========================#
#============================================================

# PIL MODIFICATIONS
# THESE WILL RETURN THE SAME DATA TYPE AS WAS GIVEN
def PILSharpness(imgs, beta):
    return np.array([sharpness(i,beta) for i in imgs])
def PILContrast(imgs, alpha):
    return np.array([contrast(i,alpha) for i in imgs])

def contrast(image, alpha):
    imageToUse = image
    if(type(imageToUse) == np.ndarray):
        isArray = True
        imageToUse = OpenCV2PIL(image)

    enhancer = ImageEnhance.Contrast(imageToUse)
    imageToUse = enhancer.enhance(alpha)
    if(isArray):
        return PIL2OpenCV(imageToUse)
    return imageToUse
        


def sharpness(image, beta):
    if(type(image) == np.ndarray):
        isArray = True
        image = OpenCV2PIL(image)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(beta)
    if(isArray):
        return PIL2OpenCV(image)
    return image




# OPENCV2 MODIFICATIONS


#==== sharpness
def gaussian_edge_sharpening(imgs):
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[1] == 1)  # check the channel is 1
    kernel_sharpen = np.array([[-1, -1, -1, -1, -1], [-1, 2, 2, 2, -1],
                               [-1, 2, 8, 2, -1], [-1, 2, 2, 2, -1], [-1, -1, -1, -1, -1]]) / 8.0
    imgs_sharpened = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_sharpened[i, 0] = cv2.filter2D(
            np.array(imgs[i, 0], dtype=np.uint8), -1, kernel_sharpen)
    return imgs_sharpened


#==== histogram equalization
def histo_equalized(imgs):
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[1] == 1)  # check the channel is 1
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i, 0] = cv2.equalizeHist(
            np.array(imgs[i, 0], dtype=np.uint8))
    return imgs_equalized


# CLAHE (Contrast Limited Adaptive Histogram Equalization)
# adaptive histogram equalization is used. In this, image is divided into
# small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then
# each of these blocks are histogram equalized as usual. So in a small
# area, histogram would confine to a small region (unless there is noise).
# If noise is there, it will be amplified. To avoid this, contrast
# limiting is applied. If any histogram bin is above the specified
# contrast limit (by default 40 in OpenCV), those pixels are clipped and
# distributed uniformly to other bins before applying histogram
# equalization. After equalization, to remove artifacts in tile borders,
# bilinear interpolation is applied
def clahe_equalized(imgs):
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[1] == 1)  # check the channel is 1
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i, 0] = clahe.apply(
            np.array(imgs[i, 0], dtype=np.uint8))
    return imgs_equalized


# ===== normalize over the dataset
def dataset_normalized(imgs):
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[1] == 1)  # check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs - imgs_mean) / imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (
            np.max(imgs_normalized[i]) - np.min(imgs_normalized[i]))) * 255
    return imgs_normalized


def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[1] == 1)  # check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) *
                      255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i, 0] = cv2.LUT(np.array(imgs[i, 0], dtype=np.uint8), table)
    return new_imgs
