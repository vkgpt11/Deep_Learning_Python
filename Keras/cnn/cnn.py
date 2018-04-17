

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# Part 1 - Building the CNN

# Importing the Keras libraries and packages 

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# Initializing the CNN
classifier = Sequential()

# Step 1 - Convolution 
# 32 -> Feature Detector 
# (3,3) --> Feature Detector shape
classifier.add(Convolution2D(32,(3,3),input_shape = (64, 64, 3),activation='relu'))

# Step 2 - Pooling - > to reduce the size of feature map
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32,(3,3),input_shape = (64, 64, 3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening 
classifier.add(Flatten())

# Step 4 - Full connection 
classifier.add(Dense(units = 128,activation='relu'))

classifier.add(Dense(units = 1,activation='sigmoid'))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

# Image pre-processing: agumentation -> reducing overfittinge-processing: agumentation -> reducing overfitting 
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
rescale=1./255,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = (8000/32),
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = (2000/32),
                         workers =8)
#classifier.fit_generator(training_set,
#                                    steps_per_epoch=8000,
#                                    epochs=2,
#                                    validation_data=test_set,
#                                    validation_steps=2000)
classifier.save_weights('test.h5')
config = classifier.to_json()
open("test.json", "wb").write(config.encode())


from keras.models import Sequential, model_from_json
from keras.layers import *

config = open("test.json", "rb").read()
model = model_from_json(config.decode())
model.load_weights('test.h5')
model
