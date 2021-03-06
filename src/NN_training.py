###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - define the training
#
##################################################


import configparser

from databaseProxy import DatabaseProxy
from extract_patches import get_data_testing
from extract_patches import get_data_training
from help_functions import *
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, core, ZeroPadding2D
from keras.layers.core import Dropout, Activation
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils.visualize_util import plot
from pre_processing import my_PreProc


# Define the neural network
def get_unet(n_ch, patch_height, patch_width, N_classes):
    inputs = Input((n_ch, patch_height, patch_width))
    conv1 = Convolution2D(32, 3, 3, activation='relu',
                          border_mode='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Convolution2D(32, 3, 3, activation='relu',
                          border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #
    conv2 = Convolution2D(64, 3, 3, activation='relu',
                          border_mode='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Convolution2D(64, 3, 3, activation='relu',
                          border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #
    conv3 = Convolution2D(128, 3, 3, activation='relu',
                          border_mode='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Convolution2D(128, 3, 3, activation='relu',
                          border_mode='same')(conv3)

    up1 = merge([UpSampling2D(size=(2, 2))(conv3), conv2],
                mode='concat', concat_axis=1)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Convolution2D(64, 3, 3, activation='relu',
                          border_mode='same')(conv4)
    #
    up2 = merge([UpSampling2D(size=(2, 2))(conv4), conv1],
                mode='concat', concat_axis=1)
    conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Convolution2D(32, 3, 3, activation='relu',
                          border_mode='same')(conv5)
    #
    conv6 = Convolution2D(N_classes, 1, 1, activation='relu', border_mode='same')(conv5)
    conv6 = core.Reshape((N_classes, patch_height * patch_width))(conv6)
    conv6 = core.Permute((2, 1))(conv6)
    ############
    conv7 = core.Activation('softplus')(conv6)

    model = Model(input=inputs, output=conv7)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Define the neural network seq
def get_seq(n_ch, patch_height, patch_width, N_classes):
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2

    inputs = Input((n_ch, patch_height, patch_width))
    zero1 = ZeroPadding2D(padding=(pad, pad))(inputs)
    conv1 = Convolution2D(filter_size, kernel, kernel,
                          border_mode='valid')(zero1)
    batch1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(batch1)
    pool1 = MaxPooling2D(pool_size=(pool_size, pool_size))(act1)

    zero2 = ZeroPadding2D(padding=(pad, pad))(pool1)
    conv2 = Convolution2D(128, kernel, kernel, border_mode='valid')(zero2)
    batch2 = BatchNormalization()(conv2)
    act2 = Activation('relu')(batch2)
    pool2 = MaxPooling2D(pool_size=(pool_size, pool_size))(act2)

    zero3 = ZeroPadding2D(padding=(pad, pad))(pool2)
    conv3 = Convolution2D(256, kernel, kernel, border_mode='valid')(zero3)
    batch3 = BatchNormalization()(conv3)
    act3 = Activation('relu')(batch3)
    pool3 = MaxPooling2D(pool_size=(pool_size, pool_size))(act3)

    zero4 = ZeroPadding2D(padding=(pad, pad))(pool3)
    conv4 = Convolution2D(512, kernel, kernel, border_mode='valid')(zero4)
    batch4 = BatchNormalization()(conv4)
    act4 = Activation('relu')(batch4)

    zero5 = ZeroPadding2D(padding=(pad, pad))(act4)
    conv5 = Convolution2D(512, kernel, kernel, border_mode='valid')(zero5)
    batch5 = BatchNormalization()(conv5)
    up1 = UpSampling2D(size=(pool_size, pool_size))(batch5)

    zero6 = ZeroPadding2D(padding=(pad, pad))(up1)
    conv6 = Convolution2D(256, kernel, kernel, border_mode='valid')(zero6)
    batch6 = BatchNormalization()(conv6)
    up2 = UpSampling2D(size=(pool_size, pool_size))(batch6)

    zero7 = ZeroPadding2D(padding=(pad, pad))(up2)
    conv7 = Convolution2D(128, kernel, kernel, border_mode='valid')(zero7)
    batch7 = BatchNormalization()(conv7)
    up3 = UpSampling2D(size=(pool_size, pool_size))(batch7)

    zero8 = ZeroPadding2D(padding=(pad, pad))(up3)
    conv8 = Convolution2D(filter_size, kernel, kernel,
                          border_mode='valid')(zero8)
    batch8 = BatchNormalization()(conv8)

    conv9 = Convolution2D(N_classes, 1, 1, border_mode='valid')(batch8)
    conv9 = core.Reshape((N_classes, patch_height * patch_width))(conv9)
    conv9 = core.Permute((2, 1))(conv9)

    act = core.Activation('softmax')(conv9)

    model = Model(input=inputs, output=act)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='adadelta',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Define the neural network gnet
# you need change function call "get_unet" to "get_gnet" or "get_seq"
# before use this network
def get_gnet(n_ch, patch_height, patch_width, N_classes):
    inputs = Input((n_ch, patch_height, patch_width))
    noise1 = GaussianNoise(sigma=0.3)(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu',
                          border_mode='same')(noise1)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Convolution2D(32, 3, 3, activation='relu',
                          border_mode='same')(conv1)
    up1 = UpSampling2D(size=(2, 2))(conv1)
    #
    conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(up1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Convolution2D(16, 3, 3, activation='relu',
                          border_mode='same')(conv2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #
    conv3 = Convolution2D(32, 3, 3, activation='relu',
                          border_mode='same')(pool1)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Convolution2D(32, 3, 3, activation='relu',
                          border_mode='same')(conv3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv3)
    #
    conv4 = Convolution2D(64, 3, 3, activation='relu',
                          border_mode='same')(pool2)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Convolution2D(64, 3, 3, activation='relu',
                          border_mode='same')(conv4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv4)
    #
    conv5 = Convolution2D(128, 3, 3, activation='relu',
                          border_mode='same')(pool3)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Convolution2D(128, 3, 3, activation='relu',
                          border_mode='same')(conv5)
    #
    up2 = merge([UpSampling2D(size=(2, 2))(conv5), conv4],
                mode='concat', concat_axis=1)
    conv6 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up2)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Convolution2D(64, 3, 3, activation='relu',
                          border_mode='same')(conv6)
    #
    up3 = merge([UpSampling2D(size=(2, 2))(conv6), conv3],
                mode='concat', concat_axis=1)
    conv7 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up3)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Convolution2D(32, 3, 3, activation='relu',
                          border_mode='same')(conv7)
    #
    up4 = merge([UpSampling2D(size=(2, 2))(conv7), conv2],
                mode='concat', concat_axis=1)
    conv8 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(up4)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Convolution2D(16, 3, 3, activation='relu',
                          border_mode='same')(conv8)
    #
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv8)
    conv9 = Convolution2D(32, 3, 3, activation='relu',
                          border_mode='same')(pool4)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Convolution2D(32, 3, 3, activation='relu',
                          border_mode='same')(conv9)
    #
    conv10 = Convolution2D(N_classes, 1, 1, activation='relu',
                           border_mode='same')(conv9)
    conv10 = core.Reshape((N_classes, patch_height * patch_width))(conv10)
    conv10 = core.Permute((2, 1))(conv10)
    ############
    conv10 = core.Activation('softmax')(conv10)

    model = Model(input=inputs, output=conv10)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# ========= Load settings from Config file
config = configparser.RawConfigParser()
config.read('configuration.txt')
# Experiment name
name_experiment = config.get('experiment name', 'name')
# training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
N_classes = int(config.get('training settings', 'N_classes'))
batch_size = int(config.get('training settings', 'batch_size'))

db = DatabaseProxy(experiment_name=name_experiment, N_classes=N_classes)

# ============ Load the data and divided in patches
patches_imgs_testx, patches_masks_testx, patches_imgs_trainx, patches_masks_trainx = db.getTestAndTrainingData(
    batches=batch_size)

patches_imgs_train, patches_masks_train = get_data_training(
    p_train_imgs_original=patches_imgs_trainx,
    p_train_groudTruth=patches_masks_trainx,  # masks
    patch_height=int(config.get('data attributes', 'patch_height')),
    patch_width=int(config.get('data attributes', 'patch_width')),
    N_subimgs=int(config.get('training settings', 'N_subimgs')),
    inside_FOV=config.getboolean('training settings', 'inside_FOV'),
    # select the patches only inside the FOV  (default == True)
    Experiment=name_experiment
)
patches_imgs_test, patches_masks_test = get_data_testing(
    p_test_imgs_original=patches_imgs_testx,
    p_test_groundTruth=patches_masks_testx,  # masks
    Imgs_to_test=10,
    patch_height=int(config.get('data attributes', 'patch_height')),
    patch_width=int(config.get('data attributes', 'patch_width')),
)

# ========= Save a sample of what you're feeding to the neural network ====
N_sample = min(patches_imgs_train.shape[0], 40)
visualize(group_images(patches_imgs_train[0:N_sample, :, :, :], 5), './' + name_experiment + '/' + "sample_input_imgs",
          save=True)
visualize(group_images(masks_colorize(patches_masks_train[0:N_sample, :, :, :]), 5),
          './' + name_experiment + '/' + "sample_input_masks", save=True)

# =========== Construct and save the model arcitecture =====
n_ch = patches_imgs_train.shape[1]
patch_height = patches_imgs_train.shape[2]
patch_width = patches_imgs_train.shape[3]
model = get_unet(n_ch, patch_height, patch_width, N_classes)  # the U-net model
print("Check: final output of the network:")
print(model.output_shape)
plot(model, to_file='./' + name_experiment + '/' +
                    name_experiment + '_model.png')  # check how the model looks like
json_string = model.to_json()
open('./' + name_experiment + '/' + name_experiment +
     '_architecture.json', 'w').write(json_string)

# ============  Training ==================================
checkpointer = ModelCheckpoint(filepath='./' + name_experiment + '/' + name_experiment + '_best_weights.h5', verbose=1,
                               monitor='val_loss', mode='auto',
                               save_best_only=True)  # save at each epoch if the validation decreased

patches_masks_train = masks_Unet(
    patches_masks_train, N_classes=N_classes)  # reduce memory consumption#
model.fit(patches_imgs_train, patches_masks_train, nb_epoch=N_epochs, batch_size=batch_size,
          verbose=2, shuffle=True, validation_split=0.2, callbacks=[checkpointer])

# ========== Save and test the last model ===================
model.save_weights('./' + name_experiment + '/' +
                   name_experiment + '_last_weights.h5', overwrite=True)
# test the model
score = model.evaluate(my_PreProc(patches_imgs_test), masks_Unet(
    patches_masks_test, N_classes=N_classes), verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
