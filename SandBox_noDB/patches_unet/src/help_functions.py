import h5py
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2, ipdb
from keras.utils.np_utils import to_categorical

def load_hdf5(infile):
  with h5py.File(infile,"r") as f:  #"with" close the file after its nested commands
    return f["image"][()]

def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)

#convert RGB image in black and white
def rgb2gray(rgb):
    assert (len(rgb.shape)==4)  #4D arrays
    assert (rgb.shape[1]==3)
    bn_imgs = rgb[:,0,:,:]
    bn_imgs = np.reshape(bn_imgs,(rgb.shape[0],1,rgb.shape[2],rgb.shape[3]))
    return bn_imgs

def gray2rgb(rgb):
    assert (len(rgb.shape)==4)  #4D arrays
    assert(rgb.shape[1]==1)
    bn_imgs = np.repeat(rgb[:,:,:,:], 3, 1)
    return bn_imgs

#Single images
def PIL2OpenCV(image):
    assert(type(image) == Image or type(image) == Image.Image)
    img = np.array(image)
    img = np.transpose(img[:,:,:1], (2,0,1))
    return img
#Single images
def OpenCV2PIL(image):
    assert(type(image) == np.ndarray)
    assert(len(image.shape) == 3)
    assert(image.shape[0] == 1)
    img = np.transpose(image, (1,2,0)).repeat(3,2)
    pil_im = Image.fromarray(img.astype(np.uint8))
    return pil_im



#group a set of images row per columns
def group_images(data,per_row):
    assert data.shape[0]%per_row==0
    assert (data.shape[1]==1 or data.shape[1]==3)
    data = np.transpose(data,(0,2,3,1))  #corect format for imshow
    all_stripe = []
    for i in range(int(data.shape[0]/per_row)):
        stripe = data[i*per_row]
        for k in range(i*per_row+1, i*per_row+per_row):
            stripe = np.concatenate((stripe,data[k]),axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    for i in range(1,len(all_stripe)):
        totimg = np.concatenate((totimg,all_stripe[i]),axis=0)
    return totimg


#visualize image (as PIL image, NOT as matplotlib!)
def visualize(data,filename):
    assert (len(data.shape)==3) #height*width*channels
    img = None
    if data.shape[2]==1:  #in case it is black and white
        data = np.reshape(data,(data.shape[0],data.shape[1]))
    if np.max(data)>1:
        img = Image.fromarray(data.astype(np.uint8))   #the image is already 0-255
    else:
        img = Image.fromarray((data*255).astype(np.uint8))  #the image is between 0-1
    img.save(filename + '.png')
    return img


#prepare the mask in the right shape for the Unet
def masks_Unet(masks):
    try:
        assert (len(masks.shape)==4)  #4D arrays
    except:
        print("Got: {}\nExpected:4".format(len(masks.shape)))
        print(masks[1])
        print(masks.shape)
        raise AssertionError
    assert (masks.shape[1]==1 )  #check the channel is 1
    im_h = masks.shape[2]
    im_w = masks.shape[3]
    print(masks.shape[0])
    masks = np.reshape(masks,(masks.shape[0],im_h*im_w))
    new_masks = np.empty((masks.shape[0],im_h*im_w,5))
    labels = np.unique(masks)
    print(labels)
    labelsP = to_categorical(labels)
    l = {}
    for q, j in zip(labels, labelsP):
        l[q] = j
    
    for i in range(masks.shape[0]):
        for j in range(im_h*im_w):
            #print(new_masks.shape)
            #print(masks.shape)
            new_masks[i,j] = l[masks[i,j]]
    
    return new_masks


def pred_to_imgs(pred,mode="original"):
    assert (len(pred.shape)==3)  #3D array: (Npatches,height*width,6)
    assert (pred.shape[2]==5 )  #check the classes are 6
    pred_images = np.empty((pred.shape[0],pred.shape[1]))  #(Npatches,height*width)
    if mode=="original":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                pred_images[i,pix]=pred[i,pix].argmax()
    elif mode=="threshold":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                if pred[i,pix,1]>=0.5:
                    pred_images[i,pix]=1
                else:
                    pred_images[i,pix]=0
    else:
        print("mode " +str(mode) +" not recognized, it can be 'original' or 'threshold'")
        exit()
    pred_images = np.reshape(pred_images,(10,1,384,288))
    return pred_images
