Messing with our inputs... Changes:
use configurations.txt file again to set patch dimensions, epochs, size, and other stuff, currently using 32x32
using original pre_processesing but reduce images to a scale of 1-5
pre_proc now done in extract_batches rather than dbproxy
re-up to 5 labels, but I don't know what to do with 255 so I just gave it label 0
masks are also on a scale of 1-5
UNET mask input/model output adjusted for 5 classes (change back to 4 if you wanna keep using 4)

As you can see in config, the last model I trained used 7100 subimgs & 10 epochs because I'm running this on
my laptop (you could probably train faster if you ran it on a potato instead of my laptop's gpu) 
Compared to the 190000 patches + 150 epochs the original paper uses I would hope that params which are a bit higher will 
produce more accurate results. 

I've tried messing with the patch size params and stuff, but I don't trust anything from that to be meaningful without 
proper training. 

Testing accuracies come out to be 70-75% usually with the small training size. 
Last model was ~72% supposedly, check the performances.txt file in experiments... Can look at the sample input imgs/masks but you may have to do it locally. They look pitch black to me on github but they shouldn't be



# NorthropComp_SP2016

This is Northrop Computing, senior project 2016/2017. The Wiki contains useful resources (readings and docs) about machine learning and neural net. 

***

### Prerequisities
The neural network is developed with the Keras library, we refer to the [Keras repository](https://github.com/fchollet/keras) for the installation.

This code has been tested with Keras 1.1.0, using either Theano or TensorFlow as backend. In order to avoid dimensions mismatch, it is important to set `"image_dim_ordering": "th"` in the `~/.keras/keras.json` configuration file. If this file isn't there, you can create it. See the Keras documentation for more details.

The following dependencies are needed:
- numpy >= 1.11.1
- PIL >=1.1.7
- opencv >=2.4.10
- h5py >=2.6.0
- configparser >=3.5.0b2
- scikit-learn >= 0.17.1



### Training


After all the parameters have been configured, you can train the neural network with:
```
python run_training.py
```
If available, a GPU will be used.
The following files will be saved in the folder with the same name of the experiment:
- model architecture (json)
- picture of the model structure (png)
- a copy of the configuration file
- model weights at last epoch (HDF5)
- model weights at best epoch, i.e. minimum validation loss (HDF5)


### Evaluate the trained model
The performance of the trained model is evaluated against the DRIVE testing dataset, consisting of 20 images (as many as in the training set).

The parameters for the testing can be tuned again in the `configuration.txt` file, specifically in the [testing settings] section, as described below:  
**[testing settings]**  
- *best_last*: choose the model for prediction on the testing dataset: best = the model with the lowest validation loss obtained during the training; last = the model at the last epoch.
- *nohup*: the standard output during the prediction is redirected and saved in a log file.

The section **[experiment name]** must be the name of the experiment you want to test, while **[data paths]** contains the paths to the testing datasets. Now the section **[training settings]** will be ignored.

Run testing by:
```
python run_testing.py
```
If available, a GPU will be used.  
The following files will be saved in the folder with same name of the experiment:
- One or more pictures including (top to bottom): original pre-processed image, ground truth, prediction
- Report on the performance
- Classification Report

##[Website](http://northropcomputing.com)
