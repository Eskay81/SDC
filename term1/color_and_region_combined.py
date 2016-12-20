import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

image  = mpimg.imread('test.jpg')

ysize = image.shape[0]
xsize = image.shape[1]
colour_select = np.copy(image)
region_select = np.copy(image)
line_image = np.copy(image)

red_threshold = 200
green_threshold = 200
blue_threshold = 200
rgb_thresholds = [red_threshold, green_threshold,blue_threshold]

left_bottom = [130, 539]
right_bottom = [810, 539]
apex = [470,310]

fit_left  = np.polyfit((left_bottom[0],apex[0]),(left_bottom[1],apex[1]),1)
fit_right = np.polyfit((right_bottom[0],apex[0]),(right_bottom[1],apex[1]),1)
fit_bottom= np.polyfit((left_bottom[0],right_bottom[0]),(left_bottom[1],right_bottom[1]),1)

colour_thresholds = (image[:,:,0] < rgb_thresholds[0]) | \
                    (image[:,:,1] < rgb_thresholds[1]) | \
                    (image[:,:,2] < rgb_thresholds[2])

[XX,YY] = np.meshgrid(np.arange(0,xsize),np.arange(0,ysize))
region_thresholds = (YY > (XX * fit_left[0]  + fit_left[1]  )) & \
                    (YY > (XX * fit_right[0] + fit_right[1] )) & \
                    (YY < (XX * fit_bottom[0]+ fit_bottom[1]))

region_select[region_thresholds] = [255,0,0]

colour_select[colour_thresholds] = [0,0,0]

line_image[~colour_thresholds & region_thresholds] = [255,0,0]

plt.subplot(221)
plt.imshow(image);plt.title("Input Image");
plt.subplot(222)
plt.imshow(colour_select);plt.title("Lane Selection");
plt.subplot(223)
plt.imshow(region_thresholds);plt.title("Region of Interest")
plt.subplot(224)
plt.imshow(line_image);plt.title("Lane in the RoI")
plt.show()
