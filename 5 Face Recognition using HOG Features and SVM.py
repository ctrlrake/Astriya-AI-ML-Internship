import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Parameters
image_size = (64, 128)
data_dir = "dataset/"

# Data & Labels
X = []
y = []

# Load and process images
for label in os.listdir(data_dir):
    label_path = os.path.join(data_dir, label)
    if not os.path.isdir(label_path):
        continue
    for file in os.listdir(label_path):
        img_path = os.path.join(label_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, image_size)

        # Extract HOG features
        features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys')
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

# Encode string labels to numeric
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train SVM
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

########################################################################################################################

def predict_face(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, image_size)
    features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys')
    pred = clf.predict([features])[0]
    name = le.inverse_transform([pred])[0]
    return name

# Example
print(predict_face("img_test.jpg"))

# --- Update these ---
data_dir = 'dataset'  # or your full path
image_size = (64, 128)

# Go through each image and visualize HoG
for person in os.listdir(data_dir):
    person_path = os.path.join(data_dir, person)
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print("Failed to load:", img_path)
            continue
        img = cv2.resize(img, image_size)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Extract HOG and visualization
        features, hog_image = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=True
        )

        # the original and HoG
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(gray, cmap='gray')
        plt.title(f"Original - {person}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(hog_image, cmap='gray')
        plt.title("HOG Features")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        # Break after showing one per class (remove if you want all)
        break

