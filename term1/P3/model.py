import os
import csv

samples = []

with open('driving_log.csv','r') as fname:
    reader = csv.reader(fname)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split

train_samples,validation_samples = train_test_split(samples, test_size = 0.1)

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import random
import cv2
from skimage.util import random_noise

def preprocess_img(input_image):
    output_image = input_image[60:140,:,:]
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2YUV)
    output_image = cv2.resize(output_image, (200,66))
    return output_image

def equalizeImage(input_image):
    rgb = cv2.cvtColor(input_image, cv2.COLOR_YUV2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
    rgb = cv2.cvtColor(hsv , cv2.COLOR_HSV2RGB)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)

batch_size = 20

nb_epoch = 30

STEERING_ANGLE_ADJ =  0.10

def generator(samples, batch_size):
    num_samples = len(samples)

    while 1:
        sklearn.utils.shuffle(samples,random_state = 10)
        for offset in range(0,num_samples, batch_size):
            batch_samples = samples[offset : offset + batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                centre_name =  '.\\IMG\\' + batch_sample[0].split('\\')[-1]
                left_name =  '.\\IMG\\' + batch_sample[1].split('\\')[-1]
                right_name =  '.\\IMG\\' + batch_sample[2].split('\\')[-1]

                centre_angle = float(batch_sample[3])

                # Centre Image and Preprocess the Image
                myImage = plt.imread(centre_name)
                myImage = preprocess_img(myImage)
                
                # append image
                images.append(myImage)
                angles.append(centre_angle)
                
                # append flipped image
                images.append(np.fliplr(myImage))
                angles.append(-centre_angle)

                #Change Contract of the Image
                # Convert from YUV to RGB and then to HSV, apply hist , then again RGB to YUB
                images.append(equalizeImage(myImage))
                angles.append(centre_angle)

                #Change Contract of the Image
                # Convert from YUV to RGB and then to HSV, apply hist , then again RGB to YUB
                images.append(equalizeImage(np.fliplr(myImage)))
                angles.append(-centre_angle)

                # Randomly add Salt and Pepper noise
                images.append(random_noise(myImage,seed = 40,mode = 's&p', amount = 0.15))
                angles.append(centre_angle)

                # Flip and Randomly add Salt and Pepper noise
                images.append(random_noise(np.fliplr(myImage),seed = 40,mode = 's&p', amount = 0.15))
                angles.append(-centre_angle)

                # LEFT IMAGE
                myImage = plt.imread(left_name)
                myImage = preprocess_img(myImage)

                # append image
                images.append(myImage)
                angles.append(centre_angle + STEERING_ANGLE_ADJ)

                # append flipped image
                images.append(np.fliplr(myImage))
                angles.append(- centre_angle - STEERING_ANGLE_ADJ)

                #Change Contract of the Image
                # Convert from YUV to RGB and then to HSV, apply hist , then again RGB to YUB
                images.append(equalizeImage(myImage))
                angles.append(centre_angle + STEERING_ANGLE_ADJ)

                #Change Contract of the Image
                # Convert from YUV to RGB and then to HSV, apply hist , then again RGB to YUB
                images.append(equalizeImage(np.fliplr(myImage)))
                angles.append(-centre_angle - STEERING_ANGLE_ADJ)

                # Randomly add Salt and Pepper noise
                images.append(random_noise(myImage,seed = 40,mode = 's&p', amount = 0.15))
                angles.append(centre_angle + STEERING_ANGLE_ADJ)

                # Flip and Randomly add Salt and Pepper noise
                images.append(random_noise(np.fliplr(myImage),seed = 40,mode = 's&p', amount = 0.15))
                angles.append(- centre_angle - STEERING_ANGLE_ADJ)

                # RIGHT IMAGE
                myImage = plt.imread(right_name)
                myImage = preprocess_img(myImage)

                # append image
                images.append(myImage)
                angles.append(centre_angle - STEERING_ANGLE_ADJ)

                # append flipped image
                images.append(np.fliplr(myImage))
                angles.append(- centre_angle + STEERING_ANGLE_ADJ)

                #Change Contract of the Image
                # Convert from YUV to RGB and then to HSV, apply hist , then again RGB to YUB
                images.append(equalizeImage(myImage))
                angles.append(centre_angle - STEERING_ANGLE_ADJ)

                #Change Contract of the Image
                # Convert from YUV to RGB and then to HSV, apply hist , then again RGB to YUB
                images.append(equalizeImage(np.fliplr(myImage)))
                angles.append(-centre_angle + STEERING_ANGLE_ADJ)

                # Randomly add Salt and Pepper noise
                images.append(random_noise(myImage,seed = 40,mode = 's&p', amount = 0.15))
                angles.append(centre_angle - STEERING_ANGLE_ADJ)

                # Flip and Randomly add Salt and Pepper noise
                images.append(random_noise(np.fliplr(myImage),seed = 40,mode = 's&p', amount = 0.15))
                angles.append(- centre_angle + STEERING_ANGLE_ADJ)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)


train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)

from keras.models import Sequential, load_model
from keras.layers import Lambda, Cropping2D
from keras.layers import Dense, Dropout, Activation, Flatten,Lambda
from keras.layers import Convolution2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import np_utils

learning_rate = 0.001

model_1 = Sequential()

model_1.add(BatchNormalization(momentum=0.9, weights=None, input_shape = (66,200,3)))

model_1.add(Convolution2D(24, 5,5, subsample = (2,2),border_mode = 'valid'))

model_1.add(Activation('elu'))

model_1.add(Convolution2D(36,5,5,subsample = (2,2),border_mode = 'valid'))

model_1.add(Activation('elu'))

model_1.add(Convolution2D(48, 5,5,subsample = (2,2),border_mode = 'valid'))

model_1.add(Activation('elu'))

model_1.add(Convolution2D(64, 3,3,subsample=(1,1),border_mode = 'valid'))

model_1.add(Activation('elu'))

model_1.add(Convolution2D(64,3,3,subsample=(1,1), border_mode = 'valid'))

model_1.add(Activation('elu'))

model_1.add(Flatten())

model_1.add(Dense(100))

model_1.add(Activation('elu'))

model_1.add(Dropout(0.2))

model_1.add(Dense(50))

model_1.add(Activation('elu'))

model_1.add(Dropout(0.2))

model_1.add(Dense(10))

model_1.add(Activation('elu'))

model_1.add(Dropout(0.2))

model_1.add(Dense(1))

model_1.add(Activation('elu'))

model_1.compile(loss = 'mse',optimizer = Adam(lr = learning_rate), metrics = ['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')

model_checkpoint = ModelCheckpoint("model.h5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period = 1)

history = model_1.fit_generator(generator(train_samples, batch_size),samples_per_epoch = 9 * len(train_samples), validation_data = generator(validation_samples, batch_size),nb_val_samples = 9 * len(validation_samples),nb_epoch = nb_epoch, verbose = 1, callbacks=[early_stopping, model_checkpoint])

model_1.save('model.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mse loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc = 'upper right')
plt.show()