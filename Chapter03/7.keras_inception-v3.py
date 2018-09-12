#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@COPIED on Mon Sep  3 23:21:51 2018
Very deep inception-v3 net used for transfer learning

Deep Learning with Keras: Implementing deep learning models and 
neural networks with the power of Python (Kindle Locations 1440-1441).

@author: rm
"""

from keras.applications.inception_v3 import InceptionV3
#from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
#from keras import backend as K
 # create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# layer.name, layer.input_shape, layer.output_shape
('mixed10', [(None, 8, 8, 320), 
             (None, 8, 8, 768), 
             (None, 8, 8, 768), 
             (None, 8, 8, 192)], 
    (None, 8, 8, 2048))
('avg_pool', (None, 8, 8, 2048), (None, 1, 1, 2048))
('flatten', (None, 1, 1, 2048), (None, 2048))
('predictions', (None, 2048), (None, 1000))

NO_OF_CLASSES_CIFAR10 = 10
NO_OF_CLASSES = NO_OF_CLASSES_CIFAR10
# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)# let's add a fully-connected layer as first layer
x = Dense(1024, activation='relu')(x)# and a logistic layer with 200 classes as last layer
predictions = Dense(NO_OF_CLASSES, activation='softmax')(x)# model to train
model = Model(input=base_model.input, output=predictions)

# that is, freeze all convolutional InceptionV3 layers
for layer in base_model.layers: 
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
 # train the model on the new data for a few epochs model.fit_generator(...)

# we chose to train the top 2 inception blocks, that is, we will freeze
 # the first 172 layers and unfreeze the rest: 
for layer in model.layers[:172]: 
    layer.trainable = False 
for layer in model.layers[172:]: 
    layer.trainable = True

# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), \
              loss='categorical_crossentropy')

## we train our model again (this time fine-tuning the top 2 inception blocks)
## alongside the top Dense layers
#model.fit_generator(...)
#
########################################################
# Experimenting for model.fit_generator(...)
from keras.preprocessing.image import ImageDataGenerator  
from keras.datasets import cifar10
from keras.utils import np_utils

import matplotlib.pyplot as plt
from datetime import datetime

BATCH_SIZE = 128
NB_EPOCH =   5           #40
NB_CLASSES = 10
VERBOSE = 1

print("Start Time: ", datetime.time(datetime.now()))
#load dataset
print("Loading CIFAR10 dataset ...")
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
 
# convert to categorical
print("Converting labels to one-hot  ...")
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES) 

# float and normalization
print("Normalize pixels to [0, 1.0] ...")
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print("Define disturbance transformations (shifts/rotates/...) to be applied to input images")
#See https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

print("Applying defined disturbances ...")
datagen.fit(X_train) #(Kindle Location 1348)

#See (Kindle Locations 1348-1349). 
# 
#https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#fit_generator
history = model.fit_generator(datagen.flow(X_train, Y_train,
                        batch_size=BATCH_SIZE),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=NB_EPOCH, 
                        verbose=VERBOSE)


print('Testing...')
score = model.evaluate(X_test, Y_test,
                     batch_size=BATCH_SIZE, verbose=VERBOSE)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])
#(Kindle Locations 1349-1351). 

#save model
model_json = model.to_json()
if(True):
    open('cifar10_inception-v3.json', 'w').write(model_json)
    model.save_weights('cifar10_weights_inception-v3.h5', overwrite=True)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
    
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
    
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print("End Time: ", datetime.time(datetime.now()))

print("\n\tDONE: ", __file__)

########################################################




