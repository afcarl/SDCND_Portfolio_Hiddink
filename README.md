[//]: # (Image References)
[loading_screen]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/Resources/Screenshots/loading_screen.png
[distortion]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/distortion.png
[distortion_theory]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/Resources/Screenshots/distortion.png
[corners_unwarp]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/corners_unwarp.png
[distortion_corrected]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/undistorted.png
[sobel_x]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/sobel_x.png
[sobel_y]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/sobel_y.png
[gradient_magnitude]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/gradient_magnitude.png
[gradient_direction]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/gradient_direction.png 
[color_thresholds]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/color_thresholds.png 
[multiple_thresholds]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/thresholded_binary.png
[region_masked]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/region_masked.png
[hough_transform]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/hough_transform.png
[perspective_transform]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/perspective_transform.png
[sliding_windows]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/sliding_windows.png
[shaded_lanes]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/shaded_lanes.png
[lane_mapping]: https://github.com/nhiddink/CarND_P4_Advanced_Lane_Finding/blob/master/resources/output_images/lane_mapping.png

![alt text][loading_screen]
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

###Camera Calibration

The code for this step is contained in Sections I & II of SDCND_P4_Hiddink.ipynb.  

![alt text][distortion]

First, I define "object points", which represent the (x, y, z) coordinates of the chessboard corners in the world. I assume that the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` is appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` is appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

![alt text][distortion_theory]

From there, I used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][corners_unwarp]

Here are the results of distortion correction on each of the test images (located in the test_images folder):

![alt text][distortion_corrected]

---

###Pipeline (Images)

![alt text][thresholded_binary]

The example above displays the results of multiple thresholds on each test image in the test_images folder. Section III of the code explains this process step-by-step, with examples of each individual threshold. In order, the thresholding used on the images is as follows:

+ Color Channel HLS & HSV Thresholding - I extract the S-channel of the original image in HLS format and combine the result with the extracted V-channel of the original image in HSV format.

![alt text][color_thresholds]

+ Binary X & Y - I use Sobel operators to filter the original image for the strongest gradients in both the x-direction and the y-direction.

![alt text][sobel_x]
![alt text][sobel_y]

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

![alt text][hough_transform]

Seeing that Hough lines were successfully created on both sides of the lane for each image, I was confident at this point that my code would be able to perform effective perspective transforms.

The code for my perspective transform is performed in a function I created called perspective_transform. The function takes in a thresholded binary image and source points, with the source points coinciding with the region masking points explained in the region masking table above. For destination points, I chose the outline of the image being transformed. Here are the results of the transforms:

![alt text][perspective_transform]

I verified that my perspective transform was working as expected by using Region masks and drawing Hough Lines. Therefore, I did not find it necessary to draw the `src` and `dst` points onto the warped test images.

The next step after transforming the perspective was to detect lane-line pixels and to fit their positions using a polynomial in Section V of my code. After developing functions for sliding_windows and shaded_lanes, I was able to detect the lanes and yield the following results:

Sliding Windows Technique:
![alt text][sliding_windows]

Shaded Lanes Technique:
![alt text][shaded_lanes]

After detecting the lanes I needed to calculate the radius of curvature for each of the polynomial fits that I performed. The results of these calculations are shown in the table below. I used the radius of curvature example code from Udacity's lessons to create the calculation cells.

| Test Image | Radius of Curvature (Left) | Radius of Curvature (Right) | 
|:----------:|:--------------------------:|:---------------------------:| 
| test1.png  | 2985.467894 meters         | 2850.142018 meters          | 
| test2.png  | 4984.505982 meters         | 12357.329365 meters         |
| test3.png  | 10088.084712 meters        | 2363.421967 meters          |
| test4.png  | 9894.520013 meters         | 2366.846436 meters          |
| test5.png  | 2548.327638 meters         | 6124.849321 meters          |
| test6.png  | 4173.313472 meters         | 45794.832663 meters         |

Another calculation performed was the offset from the lane's center. The calculations are shown in the code cell following the radius of curvature, and yielded the following:

| Test Image | Offset from Center |
|:----------:|:------------------:| 
| test1.png  | 1.020143 meters    |
| test2.png  | 0.901214 meters    |
| test3.png  | 0.935571 meters    |
| test4.png  | 1.123214 meters    |
| test5.png  | 0.930286 meters    |
| test6.png  | 1.070357 meters    |

Finally, I plotted the warped images back down onto the road such that, for each image, the lane area is identified clearly:

![alt text][lane_mapping]

---

### Pipeline (Video)

Click on the image or link below to view the video results of the project.

### P4_video_final.mp4
https://youtu.be/Vc9W7hpMgnM

<a href="http://www.youtube.com/watch?feature=player_embedded&v=Vc9W7hpMgnM
" target="_blank"><img src="http://img.youtube.com/vi/Vc9W7hpMgnM/0.jpg" 
alt="YouTube" width="240" height="180" border="10" /></a>

### 

---

### Discussion

This project was difficult in that it took a while to organize everything into a functioning system. The steps required to calibrate the camera, undistort an image and produce a thresholded binary image were straightforward. However, the perspective transform and steps following that proved to be challenging. I found the video stream was significantly dependent on the color and gradient filters applied to the images, and I had difficulty overcoming this dependancy. If my pipeline were tested on a video taken at night, I predict that it would be unable to perform correct lane detections, as the parameters I chose for the color and gradient thresholds were tailored to a video taken during the day.

### Future Plans

I feel satisfied with the results of this project. If given more time in the future, I would like to continue learning about computer vision by working on Open Source projects and extending the pipeline to other test videos:

+ I plan to build the OpenCV GameBoy Pokedex: http://www.pyimagesearch.com/2014/03/10/building-pokedex-python-getting-started-step-1-6/
+ Challenge Videos - The `challenge_video.mp4` video is an extra challenge designed to test the pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another challenge and is brutal!
+ Additional Tests - Eventually, I would like to take additional videos on my own and implement the project from scratch.
