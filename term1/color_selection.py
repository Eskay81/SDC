import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

image = mpimg.imread('test.jpg')

print('This image is : ',type(image),'with dimension',image.shape)

ysize = image.shape[0]
xsize = image.shape[1]
colour_select = np.copy(image)

red_threshold = 200
green_threshold = 200
blue_threshold = 200
rgb_threshold = [red_threshold,green_threshold,blue_threshold]

thresholds = (image[:,:,0] < rgb_threshold[0]) \
             |(image[:,:,1] < rgb_threshold[1]) \
             |(image[:,:,2] < rgb_threshold[2])

colour_select[thresholds] = [0,0,0]

plt.figure(1)
plt.subplot(211)
plt.title('Input Image')
plt.imshow(image)
plt.subplot(212)
plt.title('After thresholding')
plt.imshow(colour_select)
#plt.imshow(thresholds)
plt.show()
