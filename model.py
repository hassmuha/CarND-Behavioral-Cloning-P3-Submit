import csv
import cv2
import os

import numpy as np
import tensorflow as tf
import sklearn


# Initial Setup for Keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 2, "The number of epochs.")
flags.DEFINE_integer('batch_size', 256, "The batch size.")
flags.DEFINE_integer('correction', 0.2, "Correction for Left and Right Images")

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                image = cv2.imread(batch_sample[0])
                angle = batch_sample[1]

                images.append(image)
                angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# loading the image files
#lines = []
#with open('./data/driving_log.csv') as csvfile:
#    reader = csv.reader(csvfile)
#    for line in reader:
#        lines.append(line)
# using center right and left images
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        center_image_name = './data/IMG/'+line[0].split('/')[-1]
        left_image_name = './data/IMG/'+line[1].split('/')[-1]
        right_image_name = './data/IMG/'+line[2].split('/')[-1]
        center_angle = float(line[3])
        left_angle = center_angle + FLAGS.correction
        right_angle = center_angle - FLAGS.correction

        center_image_info = [center_image_name,center_angle]
        left_image_info = [left_image_name,left_angle]
        right_image_info = [right_image_name,right_angle]

        samples.append(center_image_info)
        samples.append(left_image_info)
        samples.append(right_image_info)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


"""images = []
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
    measurements.append(measurement)"""

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=FLAGS.batch_size)
validation_generator = generator(validation_samples, batch_size=FLAGS.batch_size)

#X_train = np.array(images)
#print(X_train)
#y_train = np.array(measurements)
#print(y_train)

input_shape = (160, 320, 3) #(160,320,3)

# Model Architecture
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(6, 5, 5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
#model.fit(X_train, y_train,FLAGS.batch_size, FLAGS.epochs, validation_split=0.2, shuffle=True)
history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=FLAGS.epochs, verbose=1)

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
