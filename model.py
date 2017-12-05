# imports
import csv
import cv2
import os

import numpy as np
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt




# Initial Setup for Keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
# Define commandline flag which can be changed while calling model.py function
flags = tf.app.flags
FLAGS = flags.FLAGS
# Model Parameters : Default values assigned but can be changed on command line access
flags.DEFINE_integer('epochs', 25, "The number of epochs.")
flags.DEFINE_integer('batch_size', 256, "The batch size.")
# Correction parameter used when left and right images are also considered for training
flags.DEFINE_integer('correction', 0.2, "Correction for Left and Right Images")

''' Generator function used to shuffle and reading only images in batch
    instead of reading all the training or validation images at once
    for memory optimization reason'''
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                # check made if we need to flip the center image for
                if batch_sample[2] == 0:
                    # loading image file
                    image = cv2.imread(batch_sample[0])
                    # color space conversion as drive.py only loads images in RGB format
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    # loading image file
                    image = cv2.imread(batch_sample[0])
                    # color space conversion as drive.py only loads images in RGB format
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # flipping the image
                    image = cv2.flip(image,1)
                angle = batch_sample[1]
                images.append(image)
                angles.append(angle)

            # For the current Batch Input and actual desired output matrix is prepared
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# Read the CSV file and load all the samples images, it contains center, left, right and flipped center images
# the steering angle is also extracted and corrected in left,right images with perspective of center image.
# Moreover for also corrected for the case of flipped center images
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    # reading each line of csv file
    for line in reader:
        #filtering to identify the center, left and right image
        center_image_name = './data/IMG/'+line[0].split('/')[-1]
        left_image_name = './data/IMG/'+line[1].split('/')[-1]
        right_image_name = './data/IMG/'+line[2].split('/')[-1]
        #apply steering angle correction for right and left image in perspective of center image
        center_angle = float(line[3])
        left_angle = center_angle + FLAGS.correction
        right_angle = center_angle - FLAGS.correction

        center_image_info = [center_image_name,center_angle,0]
        # apply steering angle correction for flipped version of center image
        center_flip_image_info = [center_image_name,center_angle*-1.0,1]
        left_image_info = [left_image_name,left_angle,0]
        right_image_info = [right_image_name,right_angle,0]

        samples.append(center_image_info)
        samples.append(center_flip_image_info)
        samples.append(left_image_info)
        samples.append(right_image_info)

# splitting dataset samples into train and validation set
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=FLAGS.batch_size)
validation_generator = generator(validation_samples, batch_size=FLAGS.batch_size)
#image shape defined
input_shape = (160, 320, 3) #(160,320,3)

# Model Architecture
model = Sequential()
# 1- Data normalization to -0.5 to 0.5 range
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
# 2- Cropping Image to include the only the view relavant for making right steering decision
model.add(Cropping2D(cropping=((70,25),(0,0))))
# 3- Layer 1 : Convolution Layer with relu activaiton function, kernel size 5 x 5 with stride of 2 x 2 and output depth of 24
model.add(Convolution2D(24, 5, 5,subsample=(2,2),activation='relu'))
# 4- Layer 2 : Convolution Layer with relu activaiton function, kernel size 5 x 5 with stride of 2 x 2 and output depth of 36
model.add(Convolution2D(36, 5, 5,subsample=(2,2),activation='relu'))
# 5- Layer 3 : Convolution Layer with relu activaiton function, kernel size 5 x 5 with stride of 2 x 2 and output depth of 48
model.add(Convolution2D(48, 5, 5,subsample=(2,2),activation='relu'))
# Dropout to avoid overfitting during training phase
model.add(Dropout(0.25))
# 6- Layer 4 : Convolution Layer with relu activaiton function, kernel size 3 x 3 with stride of 2 x 2 and output depth of 64
model.add(Convolution2D(64, 3, 3,activation='relu'))
# 7- Layer 5 : Convolution Layer with relu activaiton function, kernel size 3 x 3 with stride of 2 x 2 and output depth of 64
model.add(Convolution2D(64, 3, 3,activation='relu'))
# Dropout to avoid overfitting during training phase
model.add(Dropout(0.25))
# Flatten the input before presenting to fully connected layers
model.add(Flatten())
# 8- Layer 6 : Fully connected layer with relu activation function, and output size of 100
model.add(Dense(100, activation='relu'))
# Dropout to avoid overfitting during training phase
model.add(Dropout(0.5))
# 9- Layer 7 : Fully connected layer with relu activation function, and output size of 50
model.add(Dense(50, activation='relu'))
# Dropout to avoid overfitting during training phase
model.add(Dropout(0.5))
# 10- Layer 8 : Fully connected layer with relu activation function, and output size of 10
model.add(Dense(10, activation='relu'))
# Dropout to avoid overfitting during training phase
model.add(Dropout(0.25))
# 11- Layer 9 : Fully connected layer with relu activation function, and output size of 1 (predicted steering angle)
model.add(Dense(1))
# 12- To solve the Regression problem learning process is defined with adam optimizer to optimize Mean square error function as a loss function
model.compile(optimizer='adam', loss='mse')

# Generator is used to generate data for each batch in training as well as validation phase
history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=FLAGS.epochs, verbose=1)
# model is at the end saved to be used by other functions eg. drive.py
model.save('model.h5')

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
