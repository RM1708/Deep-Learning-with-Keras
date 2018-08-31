'''
Sub-Chapter: Recognizing CIFAR-10 images with deep learning

Gulli, Antonio. Deep Learning with Keras: Implementing deep learning 
... (Kindle Locations 1270-1271). Packt Publishing. Kindle Edition. 

env tensorflow: OK. See output snip below.
env OpenCV:
env keras:
    
@COMMENTS:
    
@SAMPLE OUTPUT:
       Recognizing CIFAR-10 images ...

X_train shape: (50000, 32, 32, 3)
50000 train samples
10000 test samples
Specifying the network ...
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 32, 32, 32)        896       
_________________________________________________________________
activation_1 (Activation)    (None, 32, 32, 32)        0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 16, 16, 32)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 16, 16, 32)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 8192)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 512)               4194816   
_________________________________________________________________
activation_2 (Activation)    (None, 512)               0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                5130      
_________________________________________________________________
activation_3 (Activation)    (None, 10)                0         
=================================================================
Total params: 4,200,842
Trainable params: 4,200,842
Non-trainable params: 0
_________________________________________________________________

Compiling model ...

Training the model ...
Train on 40000 samples, validate on 10000 samples
Epoch 1/20
40000/40000 [==============================] - 226s 6ms/step - loss: 1.7650 - acc: 0.3853 - val_loss: 1.3785 - val_acc: 0.5199
Epoch 2/20
40000/40000 [==============================] - 213s 5ms/step - loss: 1.3600 - acc: 0.5173 - val_loss: 1.2507 - val_acc: 0.5673
...
...
...
Epoch 17/20
40000/40000 [==============================] - 214s 5ms/step - loss: 0.6022 - acc: 0.7935 - val_loss: 1.0236 - val_acc: 0.6798
Epoch 18/20
40000/40000 [==============================] - 214s 5ms/step - loss: 0.5854 - acc: 0.7991 - val_loss: 0.9967 - val_acc: 0.6792
Epoch 19/20
40000/40000 [==============================] - 213s 5ms/step - loss: 0.5571 - acc: 0.8082 - val_loss: 1.1006 - val_acc: 0.6500
Epoch 20/20
40000/40000 [==============================] - 268s 7ms/step - loss: 0.5381 - acc: 0.8172 - val_loss: 1.1134 - val_acc: 0.6689

Testing the model ...
10000/10000 [==============================] - 17s 2ms/step

Test score: 1.0976396728515625
Test accuracy: 0.6677

Saving model weights ...
dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])
...


'''

from keras.datasets import cifar10 #(Kindle Location 1277).
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop

import matplotlib.pyplot as plt

from datetime import datetime

#from quiver_engine import server
# CIFAR_10 is a set of 60K images 32x32 pixels on 3 channels
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32

#constant
BATCH_SIZE = 128
NB_EPOCH = 20        #Original 20
NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIM = RMSprop()
#OPTIM = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

OPTIM_SEL = "ADAM" # "RMS", "SGD", "ADAM"
if("SGD" == OPTIM_SEL):
    OPTIM = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
elif("ADAM" == OPTIM_SEL):
    OPTIM = Adam()   
else:
    OPTIM = RMSprop()


print("\n\tStart time: ", datetime.time(datetime.now()), "\n")

print("\n\tLoading CIFAR-10 images ...\n")

#load dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
 
# convert to categorical
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES) 

# float and normalization
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# network
'''
The network constructed below is:
    
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 32, 32, 32)        896   (Conv window ?)    
_________________________________________________________________
activation_1 (Activation)    (None, 32, 32, 32)        0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 16, 16, 32)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 16, 16, 32)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 8192)              0       (16 * 16 * 32)  
_________________________________________________________________
dense_1 (Dense)              (None, 512)               4194816   = 8192 * 512
_________________________________________________________________
activation_2 (Activation)    (None, 512)               0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                5130      
_________________________________________________________________
activation_3 (Activation)    (None, 10)                0         
=================================================================
Total params: 4,200,842
Trainable params: 4,200,842 <-- (8192 * 512) + 896 + 5130
Non-trainable params: 0


'''
print("Specifying the network ...")
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
  
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dropout(0.5))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))

model.summary()


# train
print("Optimizer to be used: ", OPTIM_SEL )

print("\nCompiling model ...")
model.compile(loss='categorical_crossentropy', optimizer=OPTIM,
	metrics=['accuracy'])

print("\nTraining the model ...") 
history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE,
	epochs=NB_EPOCH, validation_split=VALIDATION_SPLIT, 
	verbose=VERBOSE)
 
print('\nTesting the model ...')
score = model.evaluate(X_test, Y_test,
                     batch_size=BATCH_SIZE, verbose=VERBOSE)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])

#server.launch(model)


#save model
print("\nSaving model weights ...")
model_json = model.to_json()
open('cifar10_architecture.json', 'w').write(model_json)
model.save_weights('cifar10_weights.h5', overwrite=True) #(Kindle Locations 1304-1305). 

# list all data in history
print(history.history.keys())
# summarize history for(of?) accuracy
#plt.plot(mo)
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for (of?) loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print("\n\tEnd time: ", datetime.time(datetime.now()), "\n")

print("\n\tDONE: ", __file__)