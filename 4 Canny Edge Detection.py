import cv2
import numpy as np
import matplotlib.pyplot as plt

# === Step 1: Load image and convert to grayscale ===
image = cv2.imread(r"C:\Users\rakes\Downloads\cameraman.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# === Step 2: Apply Gaussian Blur with your custom kernel ===
gaussian_kernel = np.array([
    [1,  4,  7,  4, 1],
    [4, 16, 26, 16, 4],
    [7, 26, 41, 26, 7],
    [4, 16, 26, 16, 4],
    [1,  4,  7,  4, 1]
], dtype=np.float32)
gaussian_kernel /= np.sum(gaussian_kernel)

blurred_image = cv2.filter2D(gray, -1, gaussian_kernel)

# === Step 3: Compute Sobel gradients ===
grad_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)

# === Step 4: Gradient Magnitude & Direction ===
gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
gradient_magnitude = np.clip(gradient_magnitude, 0, 255)
gradient_angle = np.arctan2(grad_y, grad_x) * (180.0 / np.pi)
gradient_angle = np.mod(gradient_angle, 180)

# === Step 5: Non-Maximum Suppression ===
nms_edges = np.zeros_like(gradient_magnitude, dtype=np.uint8)
plt.show()

for i in range(1, gradient_magnitude.shape[0] - 1):
    for j in range(1, gradient_magnitude.shape[1] - 1):
        angle = gradient_angle[i, j]

        if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
            neighbors = [gradient_magnitude[i, j-1], gradient_magnitude[i, j+1]]
        elif (22.5 <= angle < 67.5):
            neighbors = [gradient_magnitude[i-1, j+1], gradient_magnitude[i+1, j-1]]
        elif (67.5 <= angle < 112.5):
            neighbors = [gradient_magnitude[i-1, j], gradient_magnitude[i+1, j]]
        else:
            neighbors = [gradient_magnitude[i-1, j-1], gradient_magnitude[i+1, j+1]]

        if gradient_magnitude[i, j] >= max(neighbors):
            nms_edges[i, j] = gradient_magnitude[i, j]
        else:
            nms_edges[i, j] = 0

# === Step 6: Double Thresholding ===
low_threshold = 220
high_threshold = 255
strong_edges = (nms_edges >= high_threshold).astype(np.uint8) * 255
weak_edges = ((nms_edges >= low_threshold) & (nms_edges < high_threshold)).astype(np.uint8) * 120

# === Step 7: Edge Tracking by Hysteresis ===
final_edges = strong_edges.copy()
for i in range(1, final_edges.shape[0] - 1):
    for j in range(1, final_edges.shape[1] - 1):
        if weak_edges[i, j] != 0:
            if (strong_edges[i-1:i+2, j-1:j+2] != 0).any():
                final_edges[i, j] = 255
            else:
                final_edges[i, j] = 0

# === Step 8: Visualization ===
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Gaussian Blurred")
plt.imshow(blurred_image, cmap='gray')
plt.axis("off")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title("Non-Max Suppressed")
plt.imshow(nms_edges, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Final Canny Edges")
plt.imshow(final_edges, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Gradient Directions")
plt.imshow(gradient_angle, cmap="hsv")
plt.axis("off")
plt.show()
