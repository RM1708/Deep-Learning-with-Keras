'''
See the chapter, "Recycling pre-built deep learning models for 
                    extracting features"
    "Deep Learning with Keras: ...", by Gulli, Antonio. 
    (Kindle Locations 1424-1425). 

env tensorflow: OK. See output below
env OpenCV:
env keras:

env tensorflow Output 
---------------------
    Constructing model from pre-trained VGG16 model ...
0 input_1 (None, 224, 224, 3)
1 block1_conv1 (None, 224, 224, 64)
2 block1_conv2 (None, 224, 224, 64)
3 block1_pool (None, 112, 112, 64)
4 block2_conv1 (None, 112, 112, 128)
5 block2_conv2 (None, 112, 112, 128)
6 block2_pool (None, 56, 56, 128)
7 block3_conv1 (None, 56, 56, 256)
8 block3_conv2 (None, 56, 56, 256)
9 block3_conv3 (None, 56, 56, 256)
10 block3_pool (None, 28, 28, 256)
11 block4_conv1 (None, 28, 28, 512)
12 block4_conv2 (None, 28, 28, 512)
13 block4_conv3 (None, 28, 28, 512)
14 block4_pool (None, 14, 14, 512)
15 block5_conv1 (None, 14, 14, 512)
16 block5_conv2 (None, 14, 14, 512)
17 block5_conv3 (None, 14, 14, 512)
18 block5_pool (None, 7, 7, 512)
19 flatten (None, 25088)
20 fc1 (None, 4096)
21 fc2 (None, 4096)
22 predictions (None, 1000)
DONE
Extracting features from block4_pool ...
DONE
Getting features from cat0.jpg ...

Shape of features computed at block4_pool stage of the model:  (1, 14, 14, 512) 
    '''
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np


# pre-built and pre-trained deep learning VGG16 model
print("Constructing model from pre-trained VGG16 model ...")
base_model = VGG16(weights='imagenet', include_top=True)
for i, layer in enumerate(base_model.layers):
	print (i, layer.name, layer.output_shape)
print("DONE")

# extract features from block4_pool block
print("Extracting features from block4_pool ...")
model = Model(inputs=base_model.input, \
              outputs=base_model.get_layer('block4_pool').output)
print("DONE")

print("Getting features from cat0.jpg ...")
img_path = "/home/rm//tmp/Images/cat0.jpg"  #'cat.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# get the features from this block
features = model.predict(x)
print("\nShape of features computed at block4_pool stage of the model: ", \
      features.shape, "\n")
print (features)

print("\n\tDONE: ", __file__)
