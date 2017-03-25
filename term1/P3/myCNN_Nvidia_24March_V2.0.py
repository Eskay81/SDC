import os
import csv

samples = []

with open('driving_log_DATA.csv','r') as fname:
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

clipping_ratio = 0.01 # 25% of the image will be randomly clipped
clipping_window = 3 # 3 X 3 patch

def image_random_clipping(img_npArray):
    img_shape = img_npArray.shape
    img_clipped = img_npArray.copy()

    for i in range(0,int(img_shape[0]*img_shape[1] * clipping_ratio/2)):
        x_rand = random.randint(0,(img_shape[0]-clipping_window)) # hard coded image height from 160 - clipping_window
        y_rand = random.randint(0,(img_shape[1]-clipping_window)) # hard coded image height from 160 - clipping_window
        img_clipped[x_rand:x_rand+clipping_window,y_rand:y_rand+clipping_window] = np.ones(shape = [clipping_window,clipping_window,3],dtype = 'uint8')*255

    for i in range(0,int(img_shape[0]*img_shape[1] * clipping_ratio/2)):
        x_rand = random.randint(0,(img_shape[0]-clipping_window)) # hard coded image height from 160 - clipping_window
        y_rand = random.randint(0,(img_shape[1]-clipping_window)) # hard coded image height from 160 - clipping_window
        img_clipped[x_rand:x_rand+clipping_window,y_rand:y_rand+clipping_window] = np.zeros(shape = [clipping_window,clipping_window,3],dtype = 'uint8')

    return img_clipped

def preprocess_img(input_image):
    output_image = input_image[60:140,:,:]
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2YUV)
    output_image = cv2.resize(output_image, (200,66))
    return output_image

batch_size = 10

nb_epoch = 10

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
                angles.append(centre_angle + 0.08)

                # append flipped image
                images.append(np.fliplr(myImage))
                angles.append(-centre_angle - 0.08)

                # RIGHT IMAGE
                myImage = plt.imread(right_name)
                myImage = preprocess_img(myImage)

                # append image
                images.append(myImage)
                angles.append(centre_angle - 0.08)

                # append flipped image
                images.append(np.fliplr(myImage))
                angles.append(-centre_angle + 0.08)


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

model_1.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None, input_shape = (66,200,3)))

#model_1.add(Lambda(lambda x : x/255.0 - 0.5, input_shape = (66,200,3) , output_shape = (66,200,3)))

model_1.add(Convolution2D(24, 5,5, border_mode = 'same'))

model_1.add(Activation('elu'))

model_1.add(Convolution2D(36,5,5,border_mode = 'same'))

model_1.add(Activation('elu'))

model_1.add(Convolution2D(48, 3,3,border_mode = 'same'))

model_1.add(Activation('elu'))

model_1.add(Convolution2D(64, 3,3,border_mode = 'same'))

model_1.add(Activation('elu'))

model_1.add(Convolution2D(64,3,3, border_mode = 'valid'))

model_1.add(Activation('elu'))

model_1.add(Flatten())

model_1.add(Dense(512))

model_1.add(Activation('elu'))

#model_1.add(Dropout(0.5))

model_1.add(Dense(32))

model_1.add(Activation('elu'))

#model_1.add(Dropout(0.5))

model_1.add(Dense(1))

model_1.add(Activation('elu'))

model_1.compile(loss = 'mse',optimizer = Adam(lr = learning_rate), metrics = ['accuracy'])

history = model_1.fit_generator(train_generator,samples_per_epoch = len(train_samples), validation_data = validation_generator,nb_val_samples = 6 * len(validation_samples),nb_epoch = nb_epoch, verbose = 1)

model_1.save('driving_log_24March_V2.0.h5')

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
