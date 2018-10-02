'''
Predicting with CIFAR-10

Gulli, Antonio. Deep Learning with Keras: Implementing deep learning 
... (Kindle Location 1358). Packt Publishing. Kindle Edition. 

env tensorflow: OKAY. Though predictions are no good. See end of file
env OpenCV:
env keras:

@COMMENT:
    1. THIS CODE NEEDED MAJOR SURGERY.
    2. 

'''
import numpy as np
import scipy.misc
from keras.models import model_from_json
from keras.optimizers import SGD

from skimage import transform

from datetime import datetime

#From https://github.com/gskielian/PNG-2-CIFAR10/blob/master/batches.meta.txt
#The file at /home/rm/.keras/datasets/cifar-10-batches-py is in Chinese(Japanese?)
cifar_10_classes = ["airplane", "automobile", "bird", "cat", "deer", \
                    "dog", "frog", "horse", "ship", "truck"]

print("Start Time: ", datetime.time(datetime.now()))

#load model
USE_MODEL_CREATED_BY_CIFAR_V1 = True
USE_AUGMENTED_MODEL = True
print("Loading model weights ...")
if(not USE_MODEL_CREATED_BY_CIFAR_V1):
    #Load model & wts created by 1.keras_CIFAR10_simple.py
    model_architecture = 'cifar10_architecture.json'
    model_weights = 'cifar10_weights.h5'
    exit_msg = "model & wts created by 1.keras_CIFAR10_simple.py"
else:
    if(USE_AUGMENTED_MODEL):
        #Load model &weights created by 2.keras_CIFAR10_V1.py with USE_AUGMENTED_MODEL
        #set to True
        model_architecture = 'cifar10_architecture_augmented_data.json'
        model_weights = 'cifar10_weights_augmented_data.h5'
        exit_msg = "model & weights created by 2.keras_CIFAR10_V1.py with USE_AUGMENTED_MODEL set to True"
    else:
        #Load model & weights created by 2.keras_CIFAR10_V1.py with USE_AUGMENTED_MODEL
        #set to False
        model_architecture = 'cifar10_architecture_NOT_augmented_data.json'
        model_weights = 'cifar10_weights_NOT_augmented_data.h5'
        exit_msg = "model & weights created by 2.keras_CIFAR10_V1.py with USE_AUGMENTED_MODEL set to True"
        
        
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)

#load images
print("Getting images to be classified ...")
img_names = [
            '/home/rm/tmp/Images/cat_hiding_face.jpeg', \
             #Above is good
             #'/home/rm/tmp/Images/cat0.jpg', \
             #Above is good
             #'/home/rm/tmp/Images/cat01.jpg', \
             #Above is ***not***good
             #
             '/home/rm/tmp/Images/dog_shepherd_down_2OClock.jpg'
             #'/home/rm/tmp/Images/dog_girl_beach.jpeg'
             #Above is ***not***good. Detected as Ship
             #'/home/rm/tmp/Images/dog_german_shepherd_down.jpeg'
             #Above is ***not***good. Detected as Bird
             #'/home/rm/tmp/Images/dog_daschund.jpeg'
             #Above is ***not***good. Detected as Deer
             #'/home/rm/tmp/Images/yellow_lab_head_Horiz11_Oclock.jpg'
             #Above is good
             ]
print("Pre-Processing ...")
#imgs = [np.transpose(scipy.misc.imresize(scipy.misc.imread(img_name), \
#                                         (32, 32)),
#                     (1, 0, 2)).astype('float32')
#           for img_name in img_names]
#
# The above single nested statement has been unrolled below.
# This has been done for ease of understanding and performing needed
# surgery.
imgs = [scipy.misc.imread(img_name) for img_name in img_names]
#imgs = scipy.misc.imresize(imgs,  (32, 32))
imgs = np.asarray(imgs)
imgs[0] = transform.resize(imgs[0],  \
                            (32, 32), \
                            anti_aliasing=True) #returns float64
imgs[1] = transform.resize(imgs[1], \
                            (32, 32), \
                            anti_aliasing=True) #returns float64
imgs[0] = np.transpose(imgs[0], (1, 0, 2)).astype('float32')
imgs[1] = np.transpose(imgs[1], (1, 0, 2)).astype('float32')

#imgs = np.array(imgs) / 255

# train
print("Compiling the model ...")
optim = SGD()
model.compile(loss='categorical_crossentropy', optimizer=optim,
	metrics=['accuracy'])
print("Estimating Category ...")
category_image_0 = model.predict_classes([np.expand_dims(imgs[0],axis=0)])
print("category_image_0: ",category_image_0)
category_image_1 = model.predict_classes([np.expand_dims(imgs[1],axis=0)])
print("category_image_1: ",category_image_1)
'''
see https://www.cs.toronto.edu/~kriz/cifar.html
Expected: 
    prediction_0: [3] - Cat
    prediction_1: [5] - dog


'''

print("End Time: ", datetime.time(datetime.now()))

print("\n\tDONE: ", __file__, "\nwith ", exit_msg )

