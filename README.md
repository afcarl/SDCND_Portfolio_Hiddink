[//]: # (Image References)

[distortion]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/distortion.png
[corners_unwarp]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/corners_unwarp.png
[distortion_corrected]: http://localhost:8888/files/resources/output_images/distortion_corrected.png 
[sobel_operators]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/sobel_operators.png 
[gradient_magnitude]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/gradient_magnitude.png
[gradient_direction]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/gradient_direction.png 
[color_thresholding]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/color_thresholding.png 
[multiple_thresholds]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/multiple_thresholds.png
[region_masked]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/region_masked.png "Region Masking Example"
[hough_lines]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/hough_lines.png "Hough Lines Example"
[perspective_transform]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/perspective_transform.png "Perspective Transform Example"
[sliding_windows]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/sliding_windows.png "Sliding Windows Example"
[shaded_lanes]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/shaded_lanes.png "Shaded Lanes Example"
[lane_mapping]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/lane_mapping.png "Lane Mapping Example"

![ScreenShot](https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/Resources/Screenshots/loading_screen.png)
## Udacity's Self-Driving Car Nanodegree Program
### Project 4 - Advanced Lane Finding

---

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

The code for this step is contained in Sections I & II of SDCND_P4_Hiddink.ipynb.  

![alt text][distortion]

First, I define "object points", which represent the (x, y, z) coordinates of the chessboard corners in the world. I assume that the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` is appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` is appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

![ScreenShot](https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/Resources/Screenshots/distortion.png)

From there, I used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][corners_unwarp]

Here are the results of distortion correction on each of the test images (located in the test_images folder):

![alt text][distortion_corrected]

---

###Pipeline (Images)

![alt text][multiple_thresholds]

The example above displays the results of multiple thresholds on each test image in the test_images folder. Section III of the code explains this process step-by-step, with examples of each individual threshold. In order, the thresholding used on the images is as follows:

+ Color Channel HLS & HSV Thresholding - I extract the S-channel of the original image in HLS format and combine the result with the extracted V-channel of the original image in HSV format.

![alt text][color_thresholding]

+ Binary X & Y - I use Sobel operators to filter the original image for the strongest gradients in both the x-direction and the y-direction.

![alt text][sobel_operators]

Next, I used techniques from "Project 1 - Finding Lane Lines" to conduct sanity checks. These techniques included Region Masking & Hough Lines, and the purpose for performing them was to ensure that the thresholding steps I took were accurate enough to yield proper perspective transforms.

The region masking results are shown below. I chose to limit the mask to a region with the following points: 

| Point       | Value                                    | 
|:-----------:|:----------------------------------------:| 
| Upper Left  | (image width x 0.4, image height x 0.65) | 
| Upper Right | (image width x 0.6, image height x 0.65) |
| Lower Right | (image width, image height)              |
| Lower Left  | (0, image height)                        |

![alt text][region_masked]

Once the regions were successfully removing background noise from the images, I used HoughLinesP and the following parameters to produce Hough lines:

| Parameter        | Value  | 
|:----------------:|:------:| 
| rho              | 1      | 
| theta            | 1/180  |
| threshold        | 70     |
| min_line_length  | 10     |
| max_line_gap     | 30     |

The results of my HoughLinesP transform are shown below:

![alt text][hough_lines]

Seeing that Hough lines were successfully created on both sides of the lane for each image, I was confident at this point that my code would be able to perform effective perspective transforms.

The code for my perspective transform is performed in a function I created called perspective_transform. The function takes in a thresholded binary image and source points, with the source points coinciding with the region masking points explained in the region masking table above. For destination points, I chose the outline of the image being transformed. Here are the results of the transforms:

![Alt Text](perspective_transform)

I verified that my perspective transform was working as expected by using Region masks and drawing Hough Lines. Therefore, I did not find it necessary to draw the `src` and `dst` points onto the warped test images.

The next step after transforming the perspective was to detect lane-line pixels and to fit their positions using a polynomial in Section V of my code. After developing functions for sliding_windows and shaded_lanes, I was able to detect the lanes and yield the following results:

![Alt Text](sliding_windows)

![Alt Text](shaded_lanes)

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (Video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

This project was difficult in that it took a while to organize everything into a functioning system. The steps required to calibrate the camera, undistort an image and produce a thresholded binary image were straightforward. However, the perspective transform and steps following that proved to be challenging. 

Once it was time to implement the pipeline on test videos, more difficulties arose. These included...

### Future Plans

I feel satisfied with the results of this project. If given more time in the future, I would like to continue learning about computer vision by working on Open Source projects and extending the pipeline to other test videos:

+ I plan to build the OpenCV GameBoy Pokedex: http://www.pyimagesearch.com/2014/03/10/building-pokedex-python-getting-started-step-1-6/
+ Challenge Videos - The `challenge_video.mp4` video is an extra challenge designed to test the pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another challenge and is brutal!
+ Additional Tests - Eventually, I would like to take additional videos on my own and implement the project from scratch.
