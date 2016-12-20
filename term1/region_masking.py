import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

image = mpimg.imread('test.jpg')

print('This image is :',type(image),'with dimension',image.shape)

ysize = image.shape[0]
xsize = image.shape[1]
region_select = np.copy(image)

left_bottom = [130, 539]
right_bottom = [810, 539]
apex = [470,320]

fit_left = np.polyfit((left_bottom[0], apex[0]),(left_bottom[1],apex[1]),1)
fit_right = np.polyfit((right_bottom[0],apex[0]),(right_bottom[1],apex[1]),1)
fit_bottom = np.polyfit((left_bottom[0],right_bottom[0]),(left_bottom[1],right_bottom[1]),1)

XX , YY = np.meshgrid(np.arange(0,xsize), np.arange(0,ysize))

region_thresholds = (YY > (XX * fit_left[0]  + fit_left[1]  )) &\
                    (YY > (XX * fit_right[0] + fit_right[1] )) &\
                    (YY < (XX * fit_bottom[0]+ fit_bottom[1]))

region_select[region_thresholds] = [255,0,0]

plt.figure(1)
plt.subplot(211)
plt.title("Input Image")
plt.imshow(image)
plt.subplot(212)
plt.title("region masked")
plt.imshow(region_select)
plt.show()
