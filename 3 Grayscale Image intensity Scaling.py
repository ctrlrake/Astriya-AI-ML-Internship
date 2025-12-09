import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread(r"C:\Users\rakes\Downloads\cameraman.jpg", cv2.IMREAD_GRAYSCALE)
dark_image=image.copy()
#dark_image=np.array(dark_image)
dark_image[dark_image<128]*=2

cv2.imshow("test",dark_image)
cv2.waitKey(0)
histogram=np.zeros(256,dtype=int)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        histogram[dark_image[i,j]]+=1

plt.figure(figsize=(8,6))
plt.title("Downscaled Histogram")
plt.plot(histogram,color="black")
plt.xlim([0,255])
plt.grid()
plt.show()

