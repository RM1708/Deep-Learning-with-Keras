#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@COPIED on Mon Sep  3 23:21:51 2018
Very deep inception-v3 net used for transfer learning

Deep Learning with Keras: Implementing deep learning models and 
neural networks with the power of Python (Kindle Locations 1440-1441).

On https://keras.io/applications/ check code under 
"Fine-tune InceptionV3 on a new set of classes"

@author: rm
"""

from keras.applications.inception_v3 import InceptionV3
#from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
#from keras import backend as K
'''
 From the book (paraphrased):
    We use a trained inception-v3; we do not include the top model because we want to fine-tune on D. 
    The top level is a dense layer with 1,024 inputs and where the last output level 
    is a softmax dense layer with 200 classes of output.
    
    Training dataset D is in a domain, different from ImageNet. (In the present
    case I am going run it on CIFAR_10)
    D has 
        1. 1,024 features in input 
            (How many does CIFAR_10 have?)
            From https://github.com/rnoxy/cifar10-cnn
                "... Each image is of the size 32x32 with RGB features, 
                so the single data sample has 32x32x3=3072 features."
        2. 200 categories (CIFAR_10 has 10) in output.


'''
from datetime import datetime
print("Start Time: ", datetime.time(datetime.now()))
# create the base pre-trained model
# See file:
# "/home/rm/anaconda3/envs/tensorflow/lib/python3.6/site-packages/" + 
#                                           "keras_applications/inception_v3.py"

base_model = InceptionV3(weights='imagenet', include_top=False)

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
#print("\nLayers in base_model to train:")
#for i, layer in enumerate(base_model.layers):
#   print(i, layer.name)

'''
From the book:
base_model.output tensor has the shape (samples, channels, rows, cols) 
for dim_ordering="th" or (samples, rows, cols, channels) for dim_ordering="tf" 
but dense needs them as (samples, channels) and 
GlobalAveragePooling2D averages across (rows, cols). 

So if you look at the last four layers (where include_top=True), 
you see these shapes:

#layer.name, layer.input_shape, layer.output_shape
('mixed10', [(None, 8, 8, 320), 
             (None, 8, 8, 768), 
             (None, 8, 8, 768), 
             (None, 8, 8, 192)], 
    (None, 8, 8, 2048))
('avg_pool', (None, 8, 8, 2048), (None, 1, 1, 2048))
('flatten', (None, 1, 1, 2048), (None, 2048))
('predictions', (None, 2048), (None, 1000))

When you do include_top=False, you are removing the last three layers 
and exposing the mixed10 layer, so the GlobalAveragePooling2D layer 
converts the (None, 8, 8, 2048) to (None, 2048), 
where each element in the (None, 2048) tensor is the average value 
for each corresponding (8, 8) subtensor in the (None, 8, 8, 2048) tensor:

'''
NO_OF_CLASSES_CIFAR10 = 10
NO_OF_INPUT_FEATURES_CIFAR10 = 2048      #3072. Where did this come from?

NO_OF_CLASSES = NO_OF_CLASSES_CIFAR10
NO_OF_INPUT_FEATURES = NO_OF_INPUT_FEATURES_CIFAR10

# add a global spatial average pooling layer
x = base_model.output
#TensorShape([Dimension(None), \
#                Dimension(None), \
#                Dimension(None), \
#                Dimension(2048)])

#convert the input to the correct shape for the dense layer to handle.
x = GlobalAveragePooling2D()(x)
#TensorShape([Dimension(None), Dimension(2048)])

# let's add a fully-connected layer as first layer
#x = Dense(1024, activation='relu')(x)
x = Dense(NO_OF_INPUT_FEATURES, activation='relu')(x)
#TensorShape([Dimension(None), Dimension(2048)])

# and a logistic layer with 200 classes as last layer
predictions = Dense(NO_OF_CLASSES, activation='softmax')(x)

# model to train
model_to_train = Model(inputs=base_model.input, outputs=predictions)
#print("\nLayers in model_to_train to train:")
#for i, layer in enumerate(model_to_train.layers):
#   print(i, layer.name)

'''
All the convolutional levels are pre-trained, so we freeze them during 
the training of the full model_to_train:

'''
# that is, freeze all convolutional InceptionV3 layers
for layer in base_model.layers: 
    layer.trainable = False

# compile the model_to_train (should be done *after* setting layers to non-trainable)
model_to_train.compile(optimizer='rmsprop', \
                       loss='categorical_crossentropy')
# train the model_to_train on the new data for a few epochs 
#model.fit_generator(...)

# we chose to train the top 2 inception blocks, that is, we will freeze
 # the first 172 layers and unfreeze the rest: 
for layer in model_to_train.layers[:249]:    #[:172]: 
    layer.trainable = False 
for layer in model_to_train.layers[249:]:       #[172:]: 
    layer.trainable = True

'''
The model_to_train is then recompiled for fine-tune optimization. 
We need to recompile the model_to_train for these modifications to take effect:

'''
# we use SGD with a low learning rate
from keras.optimizers import SGD
model_to_train.compile(optimizer=SGD(lr=0.0001, momentum=0.9), \
              loss='categorical_crossentropy')

## we train our model_to_train again (this time fine-tuning the top 2 inception blocks)
## alongside the top Dense layers
#model.fit_generator(...)
#
'''
Of course, there are many parameters to fine-tune for achieving good accuracy. 
However, we are now reusing a very large pre-trained network as a starting 
point via transfer learning. 
In doing so, we can save the need to train on our machines by 
reusing what is already available in Keras.

'''
########################################################
img_rows, img_cols = 299, 299 # Resolution of inputs
channel = 3
num_classes = 10 
batch_size = 16 
epochs = 10

#from keras.datasets import cifar10
#from keras.utils import np_utils

print("Loading CIFAR10 dataset ...")
# Load Cifar10 data. Please implement your own load_data() module for your own dataset
from load_cifar10_for_imagenet import load_cifar10_data

img_rows, img_cols = 299, 299 # Resolution of inputs
channel = 3
num_classes = 10 
batch_size = 16 
nb_epoch = 3   #10

# Load Cifar10 data. Please implement your own load_data() module for your own dataset
X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols)
print("Num of Training Data Images: {}. Num of Validation Images: {}".format(\
                      X_train.shape[0], X_valid.shape[0]))

sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
model_to_train.compile(optimizer=sgd, \
              loss='categorical_crossentropy',\
              metrics=['accuracy'])

# Start Fine-tuning
print("Fine-tunng on CIFAR10 data ...")
model_to_train.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          shuffle=True,
          verbose=1,
          validation_data=(X_valid, Y_valid),
          )
########################################################
'''
#The code below is extracted from 
#2.keras_CIFAR10_V1.py
from keras.preprocessing.image import ImageDataGenerator  

import matplotlib.pyplot as plt

BATCH_SIZE = 128
NB_EPOCH =   1 #5           #40
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
print("\nTest score:", score)
#print("\nTest score:", score[0])
#print('Test accuracy:', score[1])
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

'''

print("End Time: ", datetime.time(datetime.now()))

print("\n\tDONE: ", __file__)

########################################################




