import csv
import cv2
import os

import numpy as np
import tensorflow as tf

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 2, "The number of epochs.")
flags.DEFINE_integer('batch_size', 256, "The batch size.")

# loading the image files
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
first_line = 1
for line in lines:
    # skip first line
    if first_line:
        first_line = 0
        continue
    source_path  = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
#print(X_train)
y_train = np.array(measurements)
#print(y_train)

input_shape = X_train.shape[1:] #(160,320,3)

# Model Architecture
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))

model.add(Convolution2D(6, 5, 5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train,FLAGS.batch_size, FLAGS.epochs, validation_split=0.2, shuffle=True)

model.save('model.h5')
