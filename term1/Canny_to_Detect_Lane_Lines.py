import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
# Loading Input Image
image = mpimg.imread('exit_ramp.jpg')
# converting to gray scale
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# detecting edges using Canny
# step 1 : applying gaussian blue
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray_image,(kernel_size , kernel_size),0)

#applying high and low threshold
high_threshold = 150
low_threshold = 50
#edges = cv2.Canny(gray, low_threshold, high_threshold)
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

plt.subplot(221)
plt.imshow(image);plt.title("Input Image")
plt.subplot(222)
plt.imshow(gray_image,cmap = 'gray');plt.title("Image in gray-scale")
plt.subplot(223)
plt.imshow(blur_gray,cmap = 'gray');plt.title("After Gaussian Blur Effect")
plt.subplot(224)
plt.imshow(edges,cmap = "Greys_r");plt.title("Edge Detected")
plt.show()
