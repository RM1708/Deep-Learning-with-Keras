'''
Improving the CIFAR-10 performance with deeper a network

Gulli, Antonio. Deep Learning with Keras: Implementing deep learning 
... (Kindle Locations 1310-1311). Packt Publishing. Kindle Edition. 
env tensorflow: Goes into training. Stopped execution.
env OpenCV:
env keras:

@COMMENT:
    1. Augmentation commented out in the original code.
    2. The model is trained on unaugmented data. This is not as  shown in the
    book.
'''
from keras.preprocessing.image import ImageDataGenerator #(Kindle Locations 1331-1332). 
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential #https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop

#from quiver_engine import server

import matplotlib.pyplot as plt

from datetime import datetime

# CIFAR_10 is a set of 60K images 32x32 pixels on 3 channels
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32

#constant
BATCH_SIZE = 128
NB_EPOCH =   40           #40
NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2

OPTIM_SEL = "ADAM" # "RMS", "SGD", "ADAM"
if("SGD" == OPTIM_SEL):
    OPTIM = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
elif("ADAM" == OPTIM_SEL):
    OPTIM = Adam()   
else:
    OPTIM = RMSprop()


print("\nStart Time: ", datetime.time(datetime.now()), "\n")
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

# network
print("Sequential construction of network ...")
model = Sequential()    #https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
 
model.add(Conv2D(32, (3, 3), padding='same',
                        input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
 
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))
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

print("Optimizer to be used: ", OPTIM_SEL )
print("Compiling the model ..")
# https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#compile
model.compile( \
            loss='categorical_crossentropy', \
            # https://www.tensorflow.org/api_docs/python/tf/keras/losses/categorical_crossentropy
            # https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
            # https://keras.io/losses/#usage-of-loss-functions
              optimizer=OPTIM, \
              metrics=['accuracy']
              #https://keras.io/metrics/#available-metrics
              )

# train
print("Training ...") 
AUGMENTATION = True
if(not AUGMENTATION):
    #https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#fit
    history = model.fit(X_train, \
                        Y_train, \
                        batch_size=BATCH_SIZE, \
                        epochs=NB_EPOCH, \
                        validation_split=VALIDATION_SPLIT, \
                        verbose=VERBOSE)
else:
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
    NO_OF_TRAIN_SAMPLES = int(X_train.shape[0] * (1 - VALIDATION_SPLIT))
    history = model.fit_generator(datagen.flow(X_train[: NO_OF_TRAIN_SAMPLES], \
                                               Y_train[: NO_OF_TRAIN_SAMPLES], \
                            batch_size=BATCH_SIZE), \
                            samples_per_epoch=NO_OF_TRAIN_SAMPLES, \
                            nb_epoch=NB_EPOCH, \
                            validation_data=(X_train[NO_OF_TRAIN_SAMPLES :], \
                                               Y_train[NO_OF_TRAIN_SAMPLES :]), \
                            verbose=VERBOSE)
    
    #server.launch(model)
    

print('Testing...')
score = model.evaluate(X_test, Y_test,
                     batch_size=BATCH_SIZE, verbose=VERBOSE)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])
#(Kindle Locations 1349-1351). 

#save model
model_json = model.to_json()
if(AUGMENTATION):
    open('cifar10_architecture_augmented_data.json', 'w').write(model_json)
    model.save_weights('cifar10_weights_augmented_data.h5', overwrite=True)
else:
    open('cifar10_architecture_NOT_augmented_data.json', 'w').write(model_json)
    model.save_weights('cifar10_weights_NOT_augmented_data.h5', overwrite=True)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
if(True or (not AUGMENTATION)):
    plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
if(True or (not AUGMENTATION)):
    plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print("End Time: ", datetime.time(datetime.now()))

print("\n\tDONE: ", __file__)