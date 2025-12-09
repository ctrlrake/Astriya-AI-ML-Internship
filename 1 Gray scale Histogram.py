import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread(r"C:\Users\rakes\Downloads\cameraman.jpg", cv2.IMREAD_GRAYSCALE)
histogram = np.zeros(256,dtype=int)
height = image.shape[0]
width = image.shape[1]
for i in range(height):
    for j in range(width):
        histogram[image[i][j]] += 1
plt.figure(figsize=(8, 6))
plt.title("Grayscale Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.plot(histogram, color="black")
plt.xlim([0, 255])
plt.grid()
plt.show()
print(histogram)
