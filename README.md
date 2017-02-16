[//]: # (Image References)
[color_classification]: https://github.com/nhiddink/CarND_P5_Vehicle_Detection_and_Tracking/blob/master/resources/output_images/color_classification.png
[color_distribution_visualization]: https://github.com/nhiddink/CarND_P5_Vehicle_Detection_and_Tracking/blob/master/resources/output_images/color_distribution_visualization.png
[color_histograms_visualization]: https://github.com/nhiddink/CarND_P5_Vehicle_Detection_and_Tracking/blob/master/resources/output_images/color_histograms_visualization.png
[data_visualization]: https://github.com/nhiddink/CarND_P5_Vehicle_Detection_and_Tracking/blob/master/resources/output_images/data_visualization.png
[distortion]: https://github.com/nhiddink/CarND_P5_Vehicle_Detection_and_Tracking/blob/master/resources/screenshots/distortion.png
[gradient_visualization]: https://github.com/nhiddink/CarND_P5_Vehicle_Detection_and_Tracking/blob/master/resources/output_images/gradient_visualization.png
[hog_classification]: https://github.com/nhiddink/CarND_P5_Vehicle_Detection_and_Tracking/blob/master/resources/output_images/hog_classification.png
[hog_visualization]: https://github.com/nhiddink/CarND_P5_Vehicle_Detection_and_Tracking/blob/master/resources/output_images/hog_visualization.png
[loading_screen]: https://github.com/nhiddink/CarND_P5_Vehicle_Detection_and_Tracking/blob/master/resources/screenshots/loading_screen.png
[random_image_visualization]: https://github.com/nhiddink/CarND_P5_Vehicle_Detection_and_Tracking/blob/master/resources/output_images/random_image_visualization.png
[sliding_windows]: https://github.com/nhiddink/CarND_P5_Vehicle_Detection_and_Tracking/blob/master/resources/output_images/sliding_windows.png
[spatial_binning_visualization]: https://github.com/nhiddink/CarND_P5_Vehicle_Detection_and_Tracking/blob/master/resources/output_images/spatial_binning_visualization.png
[undistorted]: https://github.com/nhiddink/CarND_P5_Vehicle_Detection_and_Tracking/blob/master/resources/output_images/undistorted.png
[undistorted_and_warped]: https://github.com/nhiddink/CarND_P5_Vehicle_Detection_and_Tracking/blob/master/resources/output_images/undistorted_and_warped.png

[![loading_screen](https://github.com/nhiddink/CarND_P5_Vehicle_Detection_and_Tracking/blob/master/resources/screenshots/loading_screen.png)](http://www.udacity.com/drive)
## Udacity's Self-Driving Car Nanodegree Program
### Project 5 - Vehicle Detection and Tracking

---

The software pipeline in SDCND_P5_Hiddink.ipynb is written to detect vehicles in a video stream. To accomplish this, the following steps were performed:
+ **Camera Calibration** - correct distortion due to the camera lens that was used to record the test videos. 
+ **Data Visualization** - understand the labeled data set as two labeled groups, "cars" and "not-cars".
+ **Feature Extraction** - apply five techniques, including histograms of color, color distribution, spatial binning, gradient magnitude, and Histogram of Oriented Gradients (HOG), on the labeled training set of images to create a feature vector.
+ **Preprocessing Data** - normalize, randomize, and split the labeled data into a training set, a validation set, and a testing set.
+ **Training** -  train a Linear SVM classifier on the labeled training set
+ **Sliding Windows** - implement a technique to search an image for vehicles using the trained classifier, and optimize the algorithm's efficiency by limiting the search area of the image and/or using heat maps that reject outliers of the positive windows.
+ **Video** - run a function using moviepy that estimates a bounding box for detected vehicles frame by frame.

Each of these steps is described in detail below.

---

### Camera Calibration

The code for this step is contained in Section I of SDCND_P5_Hiddink.ipynb.

![undistorted]

In order to account for distortion, the camera used to record the project video and shoot the test images needs to be calibrated. To do this, a series of chessboard images were introduced that displayed varying distortion angles. 

![distortion]

First, I define "object points", which represent the (x, y, z) coordinates of the chessboard corners in the world. I assume that the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, objp is just a replicated array of coordinates, and objpoints is appended with a copy of it every time I successfully detect all chessboard corners in a test image. imgpoints is appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

![undistorted_and_warped]

From there, I used the output objpoints and imgpoints to compute the camera calibration and distortion coefficients using the cv2.calibrateCamera() function. I applied this distortion correction to the test image using the cv2.undistort() function and obtained the result shown above. After successfully warping chessboard images, I was able to apply the undistort function to the test images and videos using a pickle file that stored the distortion matrix.

---

### Data Visualization

The following is a visualization of the first 10 images in the labeled dataset:

![data_visualization]

Each image is defined with either a "car" or "not-car" label. The labeled data sets used in this project are originally from the GTI Vehicle Image Database [GTI Vehicle Image Database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), augmented by examples extracted from the project video itself. In the future, this repository will include images from the [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/) and the ecently released [Udacity Labeled Dataset](https://github.com/udacity/self-driving-car/tree/master/annotations), as described in the Future Plans section below.

---

### Feature Extraction

Several different techniques for feature extraction were used in Section 2 of  this project, including histograms of color, color distribution, spatial binning, gradient magnitude, and Histogram of Oriented Gradients (HOG). Each has its own effect on the feature vector that is produced, and when combined the techniques tend to improve the chosen classifier's performance.

**Histograms of Color**

Histograms of color are used to analyze the raw pixel intensities of each color channel as features in the image. Given an image in RGB color space, color histograms of features return a concatenated feature vector based on the number of intensity bins and pixel intensity ranges.

![color_histograms_visualization]


**Color Distribution**

Color distribution maps are useful in studying color values and locating clusters of color that correspond to objects of interest in an image. When plotting these maps in 3D, often the color distribution of objects of interest is conveniently clustered along a plane.

![color_distribution_visualization]

**Spatial Binning**

Raw pixel values are useful to include in the feature vector when the desired objects in an image remain identifiable at low image resolutions, such as a car object.

![spatial_binning_visualization]

**Gradient Magnitude**

Gradient magnitude is a technique used in previous computer vision projects (Projects 1 & 4) that applies a filter that represents the magnitude of the sobel-x and sobel-y gradients of odd-numbered pixel squares, such as 3x3 or 5x5.

![gradient_visualization]

**Histogram of Oriented Gradients (HOG)**

HOG feature extraction is the most important technique utilized in this project. The scikit-image package has a built in function to handle HOG extraction, which is tuned by parameters including orientations, pixels_per_cell, and cells_per_block.

![hog_visualization]

The final feature extraction method that was implemented includes color histograms, spatial binning, and HOG, as shown in Sections 2 & 3. For HOG, the parameters were chosen as follows:

| Parameter       | Value   |
|:---------------:|:-------:| 
| orientations    | 9       |                                                                                 
| pixels_per_cell | (16,16) |
| cells_per_block | (4,4)   |
| visualise       | True    |
| feature_vector  | False   |

I chose these parameters for HOG after trial and error on test4.jpg. As shown in the visualization above, the parameters optimize the gradients and limit false positives later in the pipeline. 

---

### Preprocessing Data

The training data was normalized, randomized, and split into training and testing sets, with 20% of the data reserved for the testing set.

---

### Training the SVC Classifier

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

---

### Sliding Windows

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

---

### Video

A link to the final video output for this project is provided below. The code pipeline performs reasonably well on the entire video.

**Video in Progress**

---

### Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

---

### Future Plans

+ Datasets - 

+ **As an optional challenge** Once you have a working pipeline for vehicle detection, add in your lane-finding algorithm from the last project to do simultaneous lane-finding and vehicle detection!

+ **If you're feeling ambitious** (also totally optional though), don't stop there!  We encourage you to go out and take video of your own, and show us how you would implement this project on a new video!
