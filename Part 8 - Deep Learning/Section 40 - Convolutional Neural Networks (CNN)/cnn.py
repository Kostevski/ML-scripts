""" CNN: Train a program to differentiate between cats and dogs """
#import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd
#from sklearn.compose import ColumnTransformer
#from sklearn.metrics import confusion_matrix
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler


# Importing the libraries
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# 1 Buildin the CNN
classifier = Sequential()

# 1.1 Building the convolution
classifier.add(Conv2D(filters=32,
                      kernel_size=[3, 3],
                      input_shape=(64, 64, 3),
                      activation='relu'
                      ))

# 1.2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

# 1.3 - Flattening
classifier.add(Flatten())

# 1.4 Create classic ANN Network
classifier.add(Dense(units=128,
                     activation='relu'
                     ))
classifier.add(Dense(units=1,
                     activation='sigmoid'
                     ))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Compiling the CNN
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)