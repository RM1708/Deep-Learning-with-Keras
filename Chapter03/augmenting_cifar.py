#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gulli, Antonio. Deep Learning with Keras: 
    Implementing deep learning models and neural networks with the 
    power of Python (Kindle Locations 1331-1334).
    Packt Publishing. Kindle Edition. 

Created on Thu Aug 30 15:07:22 2018

@author: rm
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
import numpy as np

from datetime import datetime

NUM_TO_AUGMENT=5
print("Started at: ", datetime.time(datetime.now()))
 #load dataset
print("Loading dataset ...")
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
 # augumenting
print("Augmenting training set images...")
datagen = ImageDataGenerator(
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest')
'''
The rotation_range is a value in degrees (0 - 180) for 
randomly rotating pictures. width_shift and height_shift a
re ranges for randomly translating pictures vertically 
or horizontally. zoom_range is for randomly zooming pictures. 
horizontal_flip is for randomly flipping half of the 
images horizontally. fill_mode is the strategy used for 
filling in new pixels that can appear after a rotation or a shift:
'''

#xtas, ytas = [], []
xtas = []
for i in range(X_train.shape[0]//1000):
    num_aug = 0
    x = X_train[i] # (3, 32, 32)
    x = x.reshape((1,) + x.shape) # (1, 3, 32, 32)
    for x_aug in datagen.flow(x, \
                              batch_size=1,
                              save_to_dir='preview', \
                              save_prefix='cifar', \
                              save_format='jpeg'):
        if num_aug >= NUM_TO_AUGMENT:
            break
        xtas.append(x_aug[0])
        num_aug += 1
        
print("Finished at: ", datetime.time(datetime.now()))
print("\n\tDONE: ", __file__)