'''
Very deep convolutional networks for large-scale image recognition

Gulli, Antonio. Deep Learning with Keras: Implementing deep learning 
... (Kindle Location 1370). Packt Publishing. Kindle Edition. 

env tensorflow: ModuleNotFoundError: No module named 'cv2'
env OpenCV:
env keras:

'''

#from keras import backend as K
#from keras.models import Sequential
#from keras.layers.core import Flatten, Dense, Dropout
#from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
#from keras.optimizers import SGD

import tensorflow.keras.backend as  K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPool2D, ZeroPadding2D
from tensorflow.keras.optimizers import SGD
import cv2, numpy as np

from datetime import datetime

# define a VGG16 network
#Deep Learning with Keras: 
#Implementing ...(Kindle Locations 1387-1388). 
def VGG_16(weights_path=None):
    model = Sequential()
#    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
#    See https://github.com/keras-team/keras/issues/3945 answer by
#    PonyboyYbr commented on Jan 22, 2017
    model.add(ZeroPadding2D((1,1),input_shape=(224,224,3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPool2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPool2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPool2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPool2D((2,2), strides=(2,2)))

    model.add(Flatten())

    #top layer of the VGG net
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

if __name__ == "__main__":
    print("Start time: ", datetime.time(datetime.now()))
    print("Reading Image ...")
    '''
    Recognizing cats with a VGG-16 net
    Deep Learning with Keras: Implementing deep learning models 
    and neural networks with the power of Python 
    (Kindle Location 1401).
    '''
    im = cv2.imread('/home/rm/tmp/Images/cat0.jpg')
    
    print("Preprocess image ...")
    im = cv2.resize(im, (224, 224)).astype(np.float32)
#    im = im.transpose((2,0,1))
# The above will put it into theano format. 
# Image is in tensorflow format. Let it be
    im = np.expand_dims(im, axis=0)
    #K.set_image_dim_ordering("th")
    K.set_image_data_format('channels_last')


    # Test pretrained model
#    model = VGG_16('/home/rm/.keras/models/vgg16_weights.h5')
    print("Read the pre-trained model weights ...")
    model = VGG_16('/home/rm/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5')
    print("Compiling the model ...")
    optimizer = SGD()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    print("Predicting ...")
    out = model.predict(im)
    print (np.argmax(out))
    
    print("End time: ", datetime.time(datetime.now()))
    print("\n\tDONE: ", __file__)
