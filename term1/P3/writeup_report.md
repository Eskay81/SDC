
# Behavioral Cloning


The goals / steps of this project are the following:

 - Use the simulator to collect data of good driving behavior
 - Build, a convolution neural network in Keras that predicts steering angles from images
 - Train and validate the model with a training and validation set
 - Test that the model successfully drives around track one without leaving the road
 - Summarize the results with a written report
 
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

The project information is from the [Udacity link](https://github.com/udacity/CarND-Behavioral-Cloning-P3)

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

model.py containing the script to create and train the model
drive.py for driving the car in autonomous mode
model.h5 containing a trained convolution neural network
writeup_report.md or writeup_report.pdf summarizing the results



#### 2. Submission includes functional code Using the Udacity provided simulator and drive.py file, the car can be driven autonomously around the track by executing

python drive.py model.h5

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model is based from the [Nvidia End-to-End Learning for Self-Driving Cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) , with minor changes like 'ELU' is used as activation instead of 'RELU' and use for Dropout in FNN to avoid overfitting.

Here is my model.summary output:

|Layer (type)                     |Output Shape          |Param #     |Connected to                    |
|---------------------------------|:--------------------:|:----------:|:------------------------------:|
|BatchNormalization               |(None, 66, 200, 3)    |12          |batchnormalization_input_1[0][0]|
|Convolution2D                    |(None, 31, 98, 24)    |1824        |batchnormalization_1[0][0]      |
|Activation                       |(None, 31, 98, 24)    |0           |convolution2d_1[0][0]           |
|Convolution2D                    |(None, 14, 47, 36)    |21636       |activation_1[0][0]              |
|Activation                       |(None, 14, 47, 36)    |0           |convolution2d_2[0][0]           |
|Convolution2D                    |(None,  5, 22, 48)    |43248       |activation_2[0][0]              |
|Activation                       |(None,  5, 22, 48)    |0           |convolution2d_3[0][0]           |
|Convolution2D                    |(None,  3, 20, 64)    |27712       |activation_3[0][0]              |
|Activation                       |(None,  3, 20, 64)    |0           |convolution2d_4[0][0]           |
|Convolution2D                    |(None,  1, 18, 64)    |36928       |activation_4[0][0]              |
|Activation                       |(None,  1, 18, 64)    |0           |convolution2d_5[0][0]           |
|Flatten                          |(None, 1152)          |0           |activation_5[0][0]              |
|Dense                            |(None, 100)           |115300      |dropout_1[0][0]                 |
|Activation                       |(None, 100)           |0           |dense_2[0][0]                   |
|Dropout                          |(None, 100)           |0           |activation_7[0][0]              |
|Dense                            |(None, 50)            |5050        |dropout_2[0][0]                 |
|Activation                       |(None, 50)            |0           |dense_3[0][0]                   |
|Dropout                          |(None, 50)            |0           |activation_8[0][0]              |
|Dense                            |(None, 10)            |510         |dropout_3[0][0]                 |
|Activation                       |(None, 10)            |0           |dense_4[0][0]                   |
|Dropout                          |(None, 10)            |0           |activation_9[0][0]              | 
|Dense                            |(None, 1)             |11          |dropout_4[0][0]                 |
|Activation                       |(None, 1)             |0           |dense_5[0][0]                   |
|                                 |                      |            |                                |

Total params: 252,231
Trainable params: 252,225
Non-trainable params: 6

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers (4 in total with 50% dropout percent) for every FNN(Fully connected Neural Network) in order to reduce overfitting (model.py).

#### 3. Model parameter tuning

The model used an adam optimizer with following parameters:
learning_rate = 0.001
Learning Rate was along tuned manually with different values from 0.00001 to 0.01

#### 4. Appropriate training data

1. Training data was chosen to keep the vehicle driving on the road. Udacity provided data is used and additional data is recorded for recovery and left and right images are used with compensation for the camera angle (+/-0.10).

2. The training vs Validation is 90-10 % split and data is shuffled every call.

3. for data augmentation, Images are flipped left to right and steering angle is negated for the same.Also random noise(Salt and Pepper) is added with 15% ratio. So every input is augmented and results in 6 images.

4. Image is cropped from on the top and on the bottom to eliminate the sky and car bonnet part.

Centre Image (160 x 320)![Centre Image](./centre.jpg)

Image with Random Noise(Salt and Pepper - 15%) ![Image with Salt and Pepper noise](./centre_noisy.png)

Trimmed Image (80 x 320) ![Trimmed Image](./centre_trimmed.png) 

5. Image is then resized to 66 x 200 (This is not required, but it helps to make the execution faster since the size is reduced without much loss).

6. Also images are moved form RGB to YUV plane for complying with the Nvidia paper.(Even in RGB plane it's possible to train)

7. Different batch_size and epochs were tried. Final value chosen were batch_size = 240 and epochs = 75.

8. fit_generator is used to load images into the Training model.

9. The output MSE graph is ![Graph](./11April2017_30_EPOCHS.png)

### Changes to drive.py

1. Similar to the training image, here the image is clipped from 160 x 320 to 80 x 320 and converted from RGB 2 YUV plane. (drive.py line no:65 - 68)

### Things to consider while choosing Model and during Training

1. I started with simple CNN models like LeNet5 and likewise, but the accuracy was very poor. After 30 iterations, decided to follow the Nvidia architecture. Even minor changes to architecture like adjusting the Sub-sampling/strides results in significant difference in accuracy and model training duration.


2. when it comes to steering angle histogram, i manually plotted the histogram from the CSV file and took the extremes, i.e., values > abs(0.7) and appended it twice, so that the zero steering angle is not overfitting. I tried reducing the zero steering values to adjust the histogram, but that affects the car in straight road, the car was swaying.Appending the large steering angle helps in sharp turns.
