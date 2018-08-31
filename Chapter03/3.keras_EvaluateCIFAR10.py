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

#load model
print("Loading model weights ...")
model_architecture = 'cifar10_architecture.json'
model_weights = 'cifar10_weights.h5'
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)

#load images
print("Getting images to be classified ...")
img_names = ['/home/rm/tmp/Images/cat_hiding_face.jpeg', \
             '/home/rm/tmp/Images/dog_daschund.jpeg']
print("Pre-Processing ...")
#imgs = [np.transpose(scipy.misc.imresize(scipy.misc.imread(img_name), \
#                                         (32, 32)),
#                     (1, 0, 2)).astype('float32')
#           for img_name in img_names]
#
# NOTE: The above single nested statement has been unrolled below.
# This has been done for ease of understanding and performing needed
# surgery.
#
imgs = [scipy.misc.imread(img_name) for img_name in img_names]
#imgs = scipy.misc.imresize(imgs,  (32, 32))
imgs = np.asarray(imgs)
imgs[0] = transform.resize(imgs[0],  (32, 32)) #returns float64
imgs[1] = transform.resize(imgs[1],  (32, 32)) #returns float64
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
prediction_0:  [9] # This is the category for trucks:-D
prediction_1:  [7] # This one is for horses. :-D
'''

print("\n\tDONE: ", __file__)

