import os
import csv

samples = []

with open('driving_log_DATA_W_Rec_v1.csv','r') as fname:
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

batch_size = 80

nb_epoch = 75

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

                myImage = plt.imread(centre_name)
                myImage = preprocess_img(myImage)
                
                centre_angle = float(batch_sample[3])

                # append image
                images.append(myImage)
                angles.append(centre_angle)

                # append flipped image
                images.append(np.fliplr(myImage))
                angles.append(-centre_angle)

                # LEFT IMAGE
                myImage = plt.imread(left_name)
                myImage = preprocess_img(myImage)

                # append image
                images.append(myImage)
                angles.append(centre_angle - 0.08)

                # append flipped image
                images.append(np.fliplr(myImage))
                angles.append(-centre_angle + 0.08)

                # RIGHT IMAGE
                myImage = plt.imread(right_name)
                myImage = preprocess_img(myImage)

                # append image
                images.append(myImage)
                angles.append(centre_angle + 0.08)

                # append flipped image
                images.append(np.fliplr(myImage))
                angles.append(-centre_angle - 0.08)


                #images.append(random_noise(centre_image,seed = 40,mode = 's&p', amount = 0.1))
                #angles.append(centre_angle)

                #images.append(random_noise(np.fliplr(centre_image),seed = 40,mode = 's&p', amount = 0.1))
                #angles.append(-centre_angle)                

            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)


train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)


# input image dimensions
img_rows, img_cols, ch = 160,320,3 # if more than 2 variables then use this way to define(28,)*2
## number of convolutional filters
nb_filters = 32
# size of the pooling area for max pooling
pool_size = (2,2)
# convolutional kernel size
kernel_size = (3,3)

from keras.models import Sequential, load_model
from keras.layers import Lambda, Cropping2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.utils import np_utils

learning_rate = 0.0001

model_1 = Sequential()

model_1.add(BatchNormalization(momentum=0.9, weights=None, input_shape = (66,200,3)))

#model_1.add(Lambda(lambda x : x/255.0 - 0.5, input_shape = (66,200,3) , output_shape = (66,200,3)))

model_1.add(Convolution2D(24, 5,5, subsample = (2,2),border_mode = 'valid'))

model_1.add(Activation('elu'))

model_1.add(Convolution2D(36,5,5,subsample = (2,2),border_mode = 'valid'))

model_1.add(Activation('elu'))

model_1.add(Convolution2D(48, 3,3,subsample = (2,2),border_mode = 'valid'))

model_1.add(Activation('elu'))

model_1.add(Convolution2D(64, 3,3,subsample=(1,1),border_mode = 'valid'))

model_1.add(Activation('elu'))

model_1.add(Convolution2D(64,3,3,subsample=(1,1), border_mode = 'valid'))

model_1.add(Activation('elu'))

model_1.add(Flatten())

model_1.add(Dense(100))

model_1.add(Activation('elu'))

model_1.add(Dropout(0.5))

model_1.add(Dense(50))

model_1.add(Activation('elu'))

model_1.add(Dropout(0.5))

model_1.add(Dense(10))

model_1.add(Activation('elu'))

model_1.add(Dropout(0.5))

model_1.add(Dense(1))

model_1.add(Activation('elu'))

model_1.compile(loss = 'mse',optimizer = Adam(lr = learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), metrics = ['accuracy'])

history = model_1.fit_generator(generator(train_samples, batch_size),samples_per_epoch = 3 * len(train_samples), validation_data = generator(validation_samples, batch_size),nb_val_samples = 6 * len(validation_samples),nb_epoch = nb_epoch, verbose = 1)

model_1.save('driving_log_4April_V1.0.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mse loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc = 'upper right')
plt.show()
#score = model_1.evaluate(X_test, y_test, verbose = 0)
#print('Test score :', score[0])
#print('Test accuracy :',score[1])
