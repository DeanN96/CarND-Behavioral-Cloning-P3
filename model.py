import os
import numpy as np
import csv
import cv2
import sklearn

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, Input, Lambda, SpatialDropout2D, MaxPooling2D, Cropping2D
from keras.layers.convolutional import Convolution2D
import tensorflow as tf
import pandas as pd
import matplotlib.image as mpimg

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Paths
PATH_TO_IMG = './data/IMG/'
PATH_TO_CSV = './data/driving_log.csv'

# constants 
BATCH_SIZE = 32

#read in contents of csv file 

lines = []
with open(PATH_TO_CSV) as file:
    reader = csv.reader(file)
    next(reader)
    for line in reader:
        lines.append(line)

def generator(input_data, BATCH_SIZE):
    processing_batch_size = int(BATCH_SIZE / 4)
    #processing_batch_size = BATCH_SIZE
    num_of_entries = len(input_data)
    sklearn.utils.shuffle( input_data )
    while True:
        for offset in range(0, num_of_entries, processing_batch_size):
            batch_data = input_data[offset:offset + processing_batch_size]
            image_data = []
            steering_angle = []

            for batch in batch_data:
                filename_center = PATH_TO_IMG + batch[0].split('/')[-1] 
                center_image = cv2.imread(filename_center)
                
                center_angle = float(batch[3])
                image_data.append(center_image)
                steering_angle.append(center_angle)

                filename_left = PATH_TO_IMG + batch[1].split('/')[-1]
                left_image = cv2.imread(filename_left)
                left_angle = center_angle + 0.2
                image_data.append(left_image)
                steering_angle.append(left_angle)

                filename_right = PATH_TO_IMG + batch[2].split('/')[-1]
                right_image = cv2.imread(filename_right)
                right_angle = center_angle - 0.2
                image_data.append(right_image)
                steering_angle.append(right_angle)

                center_flipped = np.copy( np.fliplr(center_image))
                angle_flipped = -center_angle

                image_data.append(center_flipped)
                steering_angle.append(angle_flipped) 
                                 
            X_train = np.array(image_data)
            y_train = np.array(steering_angle) 

            yield sklearn.utils.shuffle(np.array(image_data), np.array(steering_angle))
            
train_data, validation_data = train_test_split(lines, test_size=0.2)
train_gen_instance = generator(train_data, BATCH_SIZE = BATCH_SIZE)
validation_generator_instance = generator(validation_data, BATCH_SIZE = BATCH_SIZE)

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping = ((70,25), (0,0))))

#convolution pipeline
model.add(Convolution2D(24,5,5, activation='relu', subsample=(2,2)))
model.add(Convolution2D(36,5,5, activation='relu', subsample=(2,2)))
model.add(Convolution2D(48,3,3, activation='relu', subsample=(2,2)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))

#dense pipeline
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='tanh'))
    
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_gen_instance, samples_per_epoch=len(train_data) * 4, verbose=1, validation_data=validation_generator_instance, nb_val_samples=len(validation_data)*4, nb_epoch=3)
model.save('model.h5')



