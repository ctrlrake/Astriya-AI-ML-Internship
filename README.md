# Astriya-AI-ML-Internship
Learnt from Astriya’s R&amp;D division focusing on AI and Computer Vision, implementing image processing techniques (grayscale histogram, Gaussian/median filters, HoG + SVM, Canny edge detection) and developed prototype detection models.

## Grayscale Image Histogram 

### Project Overview
This program reads a grayscale image and manually computes its pixel-intensity histogram without using any built-in histogram functions. It visualizes the distribution of pixel intensities ranging from 0 to 255, which is fundamental in image processing tasks such as preprocessing, contrast analysis, and feature extraction.

### How the Code Works

1. Image Loading  
   The image is imported using OpenCV in grayscale mode.

2. Histogram Initialization  
   A NumPy array of size 256 is created to represent the frequency of each pixel intensity.

3. Manual Histogram Computation  
   Two nested loops iterate through each pixel in the image.  
   For every pixel encountered, the corresponding intensity bin is incremented:  
   histogram[image[i][j]] += 1

4. Visualization  
   Matplotlib is used to plot the histogram curve, providing a clear visual representation of the intensity distribution.

5. Output Display  
   The histogram array (0–255) is printed to the console for numerical analysis.


### Technologies Used
- Python  
- OpenCV  
- NumPy  
- Matplotlib  

### Input
<img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/634c0e95-9cf5-434d-b3ce-05f18c646174" />

### Output

Histogram Plot:  
<img width="600" height="450" alt="grayscale" src="https://github.com/user-attachments/assets/9053cb19-9f0f-4491-a653-3344ba20e3a0" />


Histogram Values  
The program prints the frequency of each pixel intensity value from 0 to 255.


## RGB Image Histogram 

### Project Overview
This program computes and visualizes the RGB histogram of a color image by manually counting pixel intensities for each channel (Red, Green, and Blue). Instead of using built-in histogram functions, the histogram is constructed from scratch to demonstrate how pixel-level color distributions are analyzed in image processing and computer vision.


### How the Code Works

1. Image Loading  
   The image is loaded in RGB format using OpenCV.

2. Channel Separation  
   The image is split into its three channels:  
   - Blue  
   - Green  
   - Red  

3. Histogram Initialization  
   Three NumPy arrays of size 256 are created to store the frequency of each pixel intensity for every color channel.

4. Manual Histogram Computation  
   The program iterates over every pixel in the image.  
   For each pixel (i, j):  
   - blue_hist[blue[i][j]] += 1  
   - green_hist[green[i][j]] += 1  
   - red_hist[red[i][j]] += 1  

   This manually builds the histogram distribution for all three channels.

5. Visualization  
   Matplotlib is used to plot all three histograms together:  
   - Red channel in red  
   - Green channel in green  
   - Blue channel in blue  
   The plot includes axis labels, legends, and grid for clarity.


### Technologies Used
- Python  
- OpenCV  
- NumPy  
- Matplotlib  

### Input
<img width="256" height="256" alt="baboon" src="https://github.com/user-attachments/assets/aef3c0d7-4125-403f-ab80-8156070d3a1e" />

### Output

RGB Histogram Plot:

<img width="600" height="450" alt="Figure_1" src="https://github.com/user-attachments/assets/dd50a002-78e5-4c5e-aec3-61e84d512340" />

The plot illustrates how intensities are distributed across the Red, Green, and Blue channels independently.

## Grayscale Image Intensity Scaling and Histogram 

### Project Overview
This program modifies the brightness of a grayscale image by selectively amplifying darker pixel values and then manually computes the histogram of the modified image. This demonstrates how intensity scaling affects pixel distribution and overall image contrast, which is a common preprocessing step in image processing and computer vision tasks.

### How the Code Works

1. Image Loading  
   The image is loaded in grayscale mode using OpenCV.

2. Intensity Manipulation  
   A copy of the original image is created.  
   All pixels with intensity values below 128 are doubled:  
   dark_image[dark_image < 128] *= 2  
   This enhances darker regions while preserving brighter areas.

3. Displaying the Modified Image  
   The modified image is shown using OpenCV’s imshow and waitKey functions.

4. Histogram Initialization  
   A NumPy array of size 256 is created to store the frequency of each pixel intensity.

5. Manual Histogram Computation  
   Every pixel in the modified image is iterated over:  
   histogram[dark_image[i, j]] += 1  
   This builds the histogram distribution after intensity scaling.

6. Visualization  
   Matplotlib is used to plot the histogram, illustrating changes in contrast and pixel distribution.

### Technologies Used
- Python  
- OpenCV  
- NumPy  
- Matplotlib  

### Input
<img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/634c0e95-9cf5-434d-b3ce-05f18c646174" />

### Output

<img width="379" height="385" alt="image" src="https://github.com/user-attachments/assets/ef3077df-a8a3-49dd-a6e3-9a83735f729f" />

Modified Histogram Plot:

<img width="800" height="600" alt="downhist" src="https://github.com/user-attachments/assets/c77195c2-44ef-482f-88fb-d1707e2a4d89" />


The plot visualizes how intensity scaling brightens darker areas and shifts the histogram accordingly.


## Canny Edge Detection from Scratch (Gaussian Blur, Sobel, NMS, Hysteresis)

### Project Overview
This program implements a custom version of the Canny edge detection pipeline from scratch using fundamental image processing operations. Starting from a grayscale image, it applies Gaussian smoothing, computes intensity gradients using Sobel operators, performs non-maximum suppression, applies double thresholding, and finally tracks edges using hysteresis. This end-to-end implementation helps understand how classical edge detectors work internally, beyond calling a single library function.

### How the Code Works

1. Image Loading and Grayscale Conversion  
   The input image is loaded in BGR format using OpenCV and then converted to grayscale:
   - Reduces the image to a single intensity channel.
   - Simplifies subsequent gradient and edge computations.

2. Gaussian Smoothing with Custom Kernel  
   A custom 5×5 Gaussian kernel is defined and normalized so that its elements sum to 1.  
   This kernel is then applied using cv2.filter2D:
   - Reduces high-frequency noise.
   - Smooths the image while preserving overall structure.
   - Prepares the image for more stable gradient computation.

3. Sobel Gradient Computation  
   Sobel filters are applied to the blurred image:
   - grad_x: horizontal gradient (changes along x-axis).
   - grad_y: vertical gradient (changes along y-axis).  
   These gradients capture edge strength and orientation in the image.

4. Gradient Magnitude and Direction  
   The gradient magnitude is computed as:
   - sqrt(grad_x² + grad_y²)  
   It is then clipped to the range [0, 255].  
   The gradient direction is computed using arctan2(grad_y, grad_x) and converted to degrees in the range [0, 180).  
   - Magnitude indicates edge strength.
   - Direction indicates edge orientation.

5. Non-Maximum Suppression (NMS)  
   To thin out edges and keep only the most relevant pixels, non-maximum suppression is applied:
   - For each pixel, the gradient direction is used to determine which two neighboring pixels lie along the edge direction.
   - The current pixel is kept only if its magnitude is greater than or equal to both neighbors.
   - Otherwise, it is suppressed (set to zero).  
   This step produces thin, well-localized edge candidates.

6. Double Thresholding  
   Two thresholds are defined:
   - high_threshold = 255
   - low_threshold = 220  
   Based on these:
   - Strong edges: pixels with magnitude ≥ high_threshold are set to 255.
   - Weak edges: pixels with low_threshold ≤ magnitude < high_threshold are set to an intermediate value (e.g., 120).
   - Remaining pixels are treated as non-edges.  
   This separates strong, likely-real edges from weak, uncertain ones.

7. Edge Tracking by Hysteresis  
   Hysteresis links weak edges to strong ones:
   - For each weak edge pixel, its 3×3 neighborhood is checked.
   - If any neighboring pixel is a strong edge, the weak pixel is promoted to a strong edge (255).
   - Otherwise, it is discarded.  
   This step removes isolated weak responses and preserves continuous edge structures.

8. Visualization  
   Multiple visual outputs are generated using Matplotlib:
   - Original RGB image.
   - Gaussian blurred grayscale image.
   - Non-maximum suppressed edge map.
   - Final edge map after hysteresis.
   - Gradient direction map, visualized using an HSV colormap.  
   These visualizations help interpret each stage of the edge detection pipeline.

### Technologies Used
- Python  
- OpenCV  
- NumPy  
- Matplotlib  

### Input
<img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/634c0e95-9cf5-434d-b3ce-05f18c646174" />

### Output

Original and Gaussian Blurred Images:

<img width="1000" height="500" alt="Figure_2" src="https://github.com/user-attachments/assets/9ad07c13-7399-4e93-94ed-3416588f3c90" />



<img width="1200" height="600" alt="Figure_3" src="https://github.com/user-attachments/assets/3632292a-653a-4a68-bec1-396c79f49e8a" />


## Face vs Non-Face Classification using HOG Features and SVM

### Project Overview
This project implements a classical computer vision pipeline to distinguish between face images and non-face images using Histogram of Oriented Gradients (HOG) features and a Support Vector Machine (SVM) classifier. Instead of deep learning, it relies on handcrafted gradient-based descriptors to learn a decision boundary between the two classes. The script trains the model, evaluates it on a held-out test set, and visualizes HOG features for sample images from each class.

### How the Code Works

1. Dataset Structure and Loading  
   The dataset directory is assumed to follow this structure:  
   `dataset/<class_name>/<image_files>`  
   where `<class_name>` is either:
   - `faces`
   - `non_faces`  
   The script iterates through each subfolder, loading all images and assigning labels based on the folder name.

2. Preprocessing: Grayscale Conversion and Resizing  
   Each image is:
   - Loaded using OpenCV (BGR format by default).
   - Converted to grayscale (for HOG extraction).
   - Resized to a fixed size `image_size = (64, 128)` to ensure consistent feature dimensions.

3. HOG Feature Extraction  
   For every preprocessed grayscale image, HOG features are extracted using:
   - `orientations = 9`
   - `pixels_per_cell = (8, 8)`
   - `cells_per_block = (2, 2)`
   - `block_norm = 'L2-Hys'`  
   These features encode local gradient and edge information that is effective at capturing structural differences between faces and non-faces.

4. Label Encoding  
   The string labels (`"faces"` and `"non_faces"`) are converted to numeric labels using `LabelEncoder` so they can be processed by the SVM classifier.

5. Train–Test Split  
   The dataset is split into:
   - Training set (80%)
   - Test set (20%)  
   using `train_test_split` with `random_state=42` for reproducibility. This allows for proper evaluation on unseen data.

6. SVM Training  
   An SVM classifier with a linear kernel is created and trained:
   - `clf = SVC(kernel='linear', probability=True)`  
   The classifier learns a separating hyperplane in the HOG feature space that best distinguishes faces from non-faces.

7. Model Evaluation  
   After training, predictions are made on the test data:
   - `y_pred = clf.predict(X_test)`  
   A detailed classification report is printed, including:
   - Precision  
   - Recall  
   - F1-score  
   - Support for both classes (`faces`, `non_faces`)  
   In the current configuration, the model achieves an accuracy of approximately 82% on the held-out test set.

8. Face Prediction Function  
   A helper function `predict_face(image_path)` is defined to:
   - Load an image from the given path.
   - Convert it to grayscale and resize it to `image_size`.
   - Extract HOG features using the same configuration as used during training.
   - Use the trained SVM (`clf`) to predict the class label.
   - Convert the numeric prediction back to the original string label using `LabelEncoder`.  
   This function provides a simple interface for running inference on new images once the model has been trained.

9. HOG Feature Visualization  
   For each class in the dataset (one representative image per class in this script):
   - A sample image is loaded and resized.
   - Converted to grayscale.
   - HOG features are extracted with `visualize=True`, which returns both the feature vector and a visualization image (`hog_image`).  
   A Matplotlib figure is then displayed showing:
   - Left: Original grayscale image (`faces` or `non_faces`).
   - Right: HOG feature visualization, highlighting gradient structure.  
   This helps in understanding what patterns HOG is capturing for classification.

### Technologies Used
- Python  
- OpenCV  
- NumPy  
- scikit-image (HOG feature extraction)  
- scikit-learn (SVM, LabelEncoder, train/test split, evaluation)  
- Matplotlib  

### Input

Dataset folder structure:

- `dataset/faces/*.jpg`  
- `dataset/non_faces/*.jpg`

### Output

Face image HOG visualization:

<img width="1000" height="400" alt="face" src="https://github.com/user-attachments/assets/087301a1-2a5a-4eec-90ec-3c7df90f2120" />


Non-face image HOG visualization:

<img width="1000" height="400" alt="nonface" src="https://github.com/user-attachments/assets/fa7bc34b-1e56-4de6-bdd1-507c5dde4369" />



Classification report (console output) for `faces` vs `non_faces`:

<img width="696" height="314" alt="image" src="https://github.com/user-attachments/assets/bdf0d2fe-a038-436a-b3c8-2d1105d76c9c" />


The report shows precision, recall, F1-score, and support for each class, along with the overall accuracy and macro/weighted averages.

## Face Detection using OpenCV DNN (Caffe Pretrained Model)

### Project Overview
This project performs face detection on a static image using OpenCV’s Deep Neural Network (DNN) module with a pretrained Caffe model (`res10_300x300_ssd_iter_140000.caffemodel`). The script loads an input image, runs it through a Single Shot Detector (SSD)-based face detection network, and visualizes detected faces with bounding boxes and confidence scores.

### How the Code Works

1. Model Loading  
   The function `load_model(image_path)`:
   - Loads the Caffe model architecture from `deploy.prototxt.txt`.
   - Loads the pretrained weights from `res10_300x300_ssd_iter_140000.caffemodel`.  
   These files describe a deep neural network trained specifically for frontal face detection.

2. Image Reading and Preprocessing  
   - The input image is read from `image_path` using `cv2.imread`.
   - Its height and width are obtained via `image.shape[:2]`.
   - A blob is created using `cv2.dnn.blobFromImage`:
     - The image is resized to 300×300.
     - Pixel intensities are scaled by 1.0.
     - Mean subtraction is applied using `(104.0, 177.0, 123.0)`.  
   This blob is the network input format expected by the pretrained model.

3. Forward Pass through the Network  
   - The blob is set as input using `net.setInput(blob)`.
   - A forward pass is executed using `detections = net.forward()`.  
   The `detections` array contains multiple candidate bounding boxes along with associated confidence scores.

4. Processing Detections  
   For each detection:
   - The confidence score (probability of the region being a face) is read from `detections[0, 0, i, 2]`.
   - Detections with confidence <= 0.5 are discarded as weak predictions.  
     (The threshold can be adjusted for stricter or more lenient detection.)
   - The bounding box coordinates are extracted from `detections[0, 0, i, 3:7]` and scaled by the original image width and height:
     - `(startX, startY, endX, endY)`.

5. Drawing Bounding Boxes and Confidence Scores  
   For every detection above the threshold:
   - A rectangle is drawn on the original image using `cv2.rectangle`.
   - The corresponding confidence score (in percentage) is rendered as text using `cv2.putText`, positioned near the top of the box.

6. Display Loop and Window Management  
   - The resulting image with annotations is displayed in a window titled `"Output"` inside a loop.
   - The loop continues until the user presses **Esc** (key code 27).
   - After exiting, all OpenCV windows are destroyed using `cv2.destroyAllWindows()`.

7. Command-Line Usage  
   In the `__main__` block:
   - A default image path is used for direct script execution.
   - The script then attempts to read an image path from `sys.argv[1]` for command-line usage.  
   If no valid path is provided, an error message is printed showing the expected usage format:
   - `usage: script_name.py image_path`  
   - `[ERROR]: Image not found`

### Technologies Used
- Python  
- OpenCV (DNN module)  
- NumPy  

### Input

Input image:

<img width="508.5" height="507.5" alt="image" src="https://github.com/user-attachments/assets/c2bdb11d-d560-4633-ae31-45eb9fc57f7c" />



Model files required in the working directory:
- `deploy.prototxt.txt`  
- `res10_300x300_ssd_iter_140000.caffemodel`

### Output

Detected faces with bounding boxes and confidence scores:

<img width="508.5" height="507.5" alt="image" src="https://github.com/user-attachments/assets/0981cbd0-fecd-49fb-9118-53155cdd1703" />


Each detected face region is surrounded by a red rectangle, and a label indicating the detection confidence (in percentage) is shown above or near the bounding box.


## Real-Time Face Detection using OpenCV DNN and Webcam

### Project Overview
This project performs real-time face detection using a webcam feed and OpenCV’s Deep Neural Network (DNN) module with a pretrained Caffe model (`res10_300x300_ssd_iter_140000.caffemodel`). Each frame from the live video stream is processed to detect faces, and bounding boxes with confidence scores are drawn around detected regions.

### How the Code Works

1. Model and Video Stream Initialization  
   - The Caffe model architecture is loaded from `deploy.prototxt.txt`.  
   - The pretrained weights are loaded from `res10_300x300_ssd_iter_140000.caffemodel` using `cv2.dnn.readNetFromCaffe`.  
   - A webcam video stream is started using `VideoStream(src=0).start()` from the `imutils` library.  
   - A short sleep (`time.sleep(2.0)`) allows the camera sensor to warm up.

2. Frame Acquisition and Resizing  
   - Inside the main loop, frames are continuously read from the video stream:  
     `frame = vs.read()`  
   - Each frame is resized to a maximum width of 400 pixels using `imutils.resize` to:
     - Reduce computation time.
     - Maintain a consistent input size for further processing.

3. Preprocessing: Blob Construction  
   - The frame dimensions `(h, w)` are obtained from `frame.shape[:2]`.  
   - A blob is created from the resized frame using `cv2.dnn.blobFromImage`:
     - Frame is resized to 300×300.
     - Scale factor: 1.0.
     - Mean subtraction values: `(104.0, 177.0, 123.0)`.  
   - This blob is the input format that the pretrained model expects.

4. Forward Pass and Detections  
   - The blob is passed to the network using `net.setInput(blob)`.  
   - A forward pass is executed: `detections = net.forward()`.  
   - The `detections` output contains multiple candidate bounding boxes and associated confidence scores for each frame.

5. Filtering and Bounding Box Computation  
   - For each detection, the confidence score is extracted from `detections[0, 0, i, 2]`.  
   - Detections with confidence below 0.5 are discarded to filter out weak predictions.  
   - For retained detections, bounding box coordinates are obtained from `detections[0, 0, i, 3:7]` and scaled back to the original frame size using:
     - `box * np.array([w, h, w, h])`.  
   - The resulting coordinates are converted to integers: `(startX, startY, endX, endY)`.

6. Drawing Annotations on the Frame  
   - Rectangles are drawn around detected faces using `cv2.rectangle`.  
   - Confidence scores (as percentages) are rendered above or near each bounding box using `cv2.putText` with `cv2.FONT_HERSHEY_SIMPLEX`.  
   - This provides a live visualization of both the detection location and its confidence level.

7. Display and Keyboard Handling  
   - The annotated frame is displayed in a window titled `"Frame"` using `cv2.imshow`.  
   - `cv2.waitKey(1)` is used to capture keyboard input at each iteration.  
   - When the Esc key (key code 27) is pressed:
     - The loop is terminated.
     - The video stream is stopped.

8. Cleanup  
   - All OpenCV windows are closed using `cv2.destroyAllWindows()`.  
   - The video stream is properly released using `vs.stop()`.

### Technologies Used
- Python  
- OpenCV (DNN and image processing)  
- NumPy  
- imutils (VideoStream helper)  
- time (for warm-up delay)

### Output

Real-time video feed with detected faces highlighted:



https://github.com/user-attachments/assets/0d6c3236-4380-4a7b-87e1-9a5d7a8faa37



- Each detected face is enclosed in a red bounding box.  
- A confidence score (percentage) is displayed near each bounding box, indicating the model’s confidence in the detection.









