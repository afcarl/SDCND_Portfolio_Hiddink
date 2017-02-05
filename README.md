[//]: # (Image References)

[loading_screen]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/Resources/Screenshots/loading_screen.png
[distortion]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/distortion.png "Distortion Example"
[corners_unwarp]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/corners_unwarp.png "Undistortion Example"
[sobel_operators]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/sobel_operators.png "Sobel Operators Example"
[gradient_magnitude]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/gradient_magnitude.png "Gradient Magnitude Example"
[gradient_direction]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/gradient_direction.png "Gradient Direction Example"
[color_thresholding]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/color_thresholding.png "Color Thresholding Example"
[multiple_thresholds]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/multiple_thresholds.png "Multiple Thresholds Example"
[region_masked]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/region_masked.png "Region Masking Example"
[hough_lines]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/hough_lines.png "Hough Lines Example"
[perspective_transform]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/perspective_transform.png "Perspective Transform Example"
[sliding_windows]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/sliding_windows.png "Sliding Windows Example"
[shaded_lanes]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/shaded_lanes.png "Shaded Lanes Example"
[lane_mapping]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/lane_mapping.png "Lane Mapping Example"

![Alt Text](loading_screen)
# Udacity's Self-Driving Car Nanodegree Program
## Project 4 - Advanced Lane Finding

This project utilizes several computer vision algorithms and techniques to perform advanced lane finding on test images and video streams. There are several steps involved in this process, including: 

* Computing the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Applying a distortion correction to raw images.
* Using color transforms, gradients, etc., to create a thresholded binary image.
* Appling a perspective transform to rectify binary image ("birds-eye view").
* Detecting lane pixels and fit to find the lane boundary.
* Determining the curvature of the lane and vehicle position with respect to center.
* Warping the detected lane boundaries back onto the original image.
* Outputing the visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Click on the image below to view a YouTube video showcasing the results of the project.

#####_VIDEO WORK IN PROGRESS_

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in Sections I & II of SDCND_P4_Hiddink.ipynb.  

First, I define "object points", which represent the (x, y, z) coordinates of the chessboard corners in the world. I assume that the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` is appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` is appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

![alt text][distortion]

From there, I used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][corners_unwarp]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

In this project, your goal is to write a software pipeline to identify the lane boundaries in a video, but the main output or product we want you to create is a detailed writeup of the project.  Check out the [writeup template](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup.  

Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
he images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include a description in your writeup for the project of what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on. 

Apply a distortion correction to raw images.

![ScreenShot](https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/Resources/Screenshots/distortion.png)

Use color transforms, gradients, etc., to create a thresholded binary image.
I experimented with several color transformations and gradients in my code. These included:
+ HSV Color transform using the S-channel
+ Gradient Magnitude
+ Gradient Direction


Apply a perspective transform to rectify binary image ("birds-eye view").

4-Point Perspective Transform Example:
http://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/


Detect lane pixels and fit to find the lane boundary.

Determine the curvature of the lane and vehicle position with respect to center.

![ScreenShot](https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/Resources/Screenshots/highway_specs.png)

Warp the detected lane boundaries back onto the original image.

Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Creating a great writeup:
---
A great writeup should include the rubric points as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  
Challenge Videos

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious (again, totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!



Conclusion


Future Plans

+ I plan to build the OpenCV GameBoy Pokedex: http://www.pyimagesearch.com/2014/03/10/building-pokedex-python-getting-started-step-1-6/


