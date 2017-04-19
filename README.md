# NorthropComp_SP2016

This is Northrop Computing, senior project 2016/2017. The Wiki contains useful resources (readings and docs) about machine learning and neural net. 

### Prerequisities
The neural network is developed with the Keras library, we refer to the [Keras repository](https://github.com/fchollet/keras) for the installation.

This code has been tested with Keras 1.1.0, using either Theano or TensorFlow as backend. In order to avoid dimensions mismatch, it is important to set `"image_dim_ordering": "th"` in the `~/.keras/keras.json` configuration file. If this file isn't there, you can create it. See the Keras documentation for more details.

The following dependencies are also needed:
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


##Previous Work
This is a fork of[https://github.com/orobix/retina-unet](https://github.com/orobix/retina-unet)

Most of the changes have been related to massaging our dataset into the preexisting model. 


##License
MIT 

Copyright 2017 Joseph Marylander, Thomas Lillis, Ryan Riley, Dylan McKinney, Camilla Lambrocco, Oliver Hanna

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.