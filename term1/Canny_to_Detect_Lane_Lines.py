import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

image = mpimg.imread('exit_ramp.jpg')
plt.imshow(image)
plt.show()

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt.imshow(gray,cmap = 'gray')
plt.show()
