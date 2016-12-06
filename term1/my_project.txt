import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

image = mpimg.imread('test_images/solidWhiteRight.jpg')
print('This image is :',type(image),'with width :',image.shape[1],'and height :'image.shape[0])
plt.imshow(image)
plt.show()