#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilizing Keras built-in VGG-16 net module

@COPIED on Mon Sep  3 10:46:31 2018, from:
    Gulli, Antonio. Deep Learning with Keras: 
        Implementing deep learning models and neural networks with 
        the power of Python 
        (Kindle Locations 1412-1415). 

@author: rm
"""

#from keras.models import Model
#from keras.preprocessing import image
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
import numpy as np
import cv2
 # prebuild model with pre-trained weights on imagenet
model = VGG16(weights='imagenet', include_top=True)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

# resize into VGG16 trained images' format
im = cv2.resize(cv2.imread(\
#            '/home/rm/tmp/Images/choo_choo_train_Horiz11_Oclock.jpg'\
             #'/home/rm/tmp/Images/dog_daschund.jpeg' \
             #'/home/rm/tmp/Images/dog_german_shepherd_down.jpeg'
             #'/home/rm/tmp/Images/dog_girl_beach.jpeg' #This one is totally
                                                        #at sea
             '/home/rm/tmp/Images/dog_shepherd_down_2OClock.jpg'
                           ), \
            (224, 224))
im = np.expand_dims(im, axis=0)
 # predict
out = model.predict(im)
#https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ravel.html
plt.plot(out.ravel())
plt.show()

list_of_probabilities = out[0].tolist()
index_max_probability = np.argmax(out)
max_probabilty = list_of_probabilities[index_max_probability]
print ("Index of most Probable Class: ", index_max_probability) #this should print
                                                            # 820 for steaming train
print("Prob of winning index: ", int(max_probabilty *\
                                        1.0E04)/1.0E04)

tmp = list_of_probabilities.pop(index_max_probability)
 #Sanity check
assert tmp == max_probabilty, "The max value popped from list"
index_of_runner_up =  np.argmax(list_of_probabilities)
probabilty_of_runner_up = \
    list_of_probabilities[index_of_runner_up]
print ("Index of Runner-Up Class: ", index_of_runner_up) 
print("Prob of Runner-Up index: ", int(probabilty_of_runner_up *\
                                        1.0E04)/1.0E04)

print("\n\tDONE: ", __file__)

