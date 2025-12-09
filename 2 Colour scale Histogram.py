import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread(r"C:\Users\rakes\Downloads\baboon.png")
blue,green,red = cv2.split(image)
blue_hist = np.zeros(256,dtype=int)
green_hist = np.zeros(256,dtype=int)
red_hist = np.zeros(256,dtype=int)

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        blue_hist[blue[i][j]] += 1
        green_hist[green[i][j]] += 1
        red_hist[red[i][j]] += 1
plt.figure(figsize=(8, 6))
plt.title("RGB Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.plot(red_hist, color="red",label="Red Histogram")
plt.plot(green_hist, color="green",label="Green Histogram")
plt.plot(blue_hist, color="blue",label="Blue Histogram")
plt.xlim([0, 255])
plt.legend()
plt.grid()
plt.show()
