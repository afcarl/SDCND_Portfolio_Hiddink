[//]: # (Image References)
[atom]: https://github.com/nhiddink/CarND_P7_Unscented_Kalman_Filters/blob/master/screenshots/atom.png
[loading_screen]: https://github.com/nhiddink/CarND_P7_Unscented_Kalman_Filters/blob/master/screenshots/loading_screen.png
[sensor_pros_cons]: https://github.com/nhiddink/CarND_P7_Unscented_Kalman_Filters/blob/master/screenshots/sensor_pros_cons.png
[combined result]: https://github.com/nhiddink/CarND_P7_Unscented_Kalman_Filters/blob/master/screenshots/combined_result.png
[lidar result]: https://github.com/nhiddink/CarND_P7_Unscented_Kalman_Filters/blob/master/screenshots/lidar_result.png
[radar result]: https://github.com/nhiddink/CarND_P7_Unscented_Kalman_Filters/blob/master/screenshots/radar_result.png
[ctrv model]: https://github.com/nhiddink/CarND_P7_Unscented_Kalman_Filters/blob/master/screenshots/ctrv_model.png
[ukf goal]: https://github.com/nhiddink/CarND_P7_Unscented_Kalman_Filters/blob/master/screenshots/ukf_goal.png

[![loading_screen](https://github.com/nhiddink/CarND_P5_Vehicle_Detection_and_Tracking/blob/master/resources/screenshots/loading_screen.png)](http://www.udacity.com/drive)

## Udacity's Self-Driving Car Nanodegree Program
### Project 7 - Unscented Kalman Filters

---

The starter code and input data for this project was taken from the following repository on GitHub: 
https://github.com/udacity/CarND-Unscented-Kalman-Filter-Project

---

## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

---

## Basic Build Instructions

[![atom](https://github.com/nhiddink/CarND_P6_Extended_Kalman_Filters/blob/master/screenshots/atom.png)](https://github.com/atom/atom/releases)

GitHub's Atom editor is a useful tool for working on the code files in this repository. Click on the image above to visit the download page for Atom 1.15. When you have finished editing the source code, build the project by performing these steps:

1. Inside of your project's directory, make a build directory: `mkdir build && cd build`
2. Compile your source code: `cmake .. && make`
3. Run the program in the Terminal window, within the build directory: `./UnscentedKF path/to/input.txt path/to/output.txt`. You can find
   some sample inputs in 'data/'.
    - eg. `./UnscentedKF ../data/sample-laser-radar-measurement-data-1.txt output.txt`

To run the data visualization 2d simulator:
1. Ensure that the kalman_tracker_mac app and kalman_tracker.py script files are in your build directory.
2. Open the app, and select your desired windowed application size, then click "Play!".
3. Select the sensor types to include in the simulation, and select your build folder for the data storage location.
4. In the Terminal, enter **python kalman_tracker.py ./UnscentedKF** and click "Run" on the simulator to begin the simulation.

---

## Simulation Results

The following results were produced when running the UKF source code in this repository through the kalman_tracker.py script:

**Sensor Fusion Result**
![combined result]

**Radar-only Result**
![radar result]

**Lidar-only Result**
![lidar result]

---

## Code Style

In general, this repository conforms to [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html).

---

## Overview

This project is an implementation of an Unscented Kalman Filter in C++. The filter utilizes both laser and radar data inputs, as each has its own advantages. Laser data can accurately define an object's position in space. Radar data provides valuable speed data that can be used to track an object's velocity relative to the sensor. The following schematic outlines these concepts:

![sensor_pros_cons]

Combining data from multiple sensors such that the resulting information has less uncertainty than either of the originals alone is the essence of [Sensor Fusion](https://en.wikipedia.org/wiki/Sensor_fusion).

One limitation of the Extented Kalman Filter as compared to the Unscented Kalman Filter is that it does not approximate well for turning vehicles. The Unscented Kalman filter uses the CTRV model to handle this.

![ctrv model]

The goal of the Unscented Kalman Filter is to approximate a Gaussian distribution of the CTRV model using sigma points, or points that are specifically chosen to model an approximation of a non-linear function.

![ukf goal]

Once sigma points are generated using educated predictions, the kalman filter updates the output data and the process repeats for the total number of measurements.

---

## Results (Terminal)

The following results for Root Mean Square Error (RMSE) Accuracy were produced for each set of data, respectively:

**sample-laser-radar-measurement-data-1.txt**

Total Measurements: 1224

| Variable | RMSE |
|:--------:|:----:|
| px       | 0.0741626 |
| py       | 0.0831793 |
| vx       | 0.572394  | 
| vy       | 0.578567  |

**sample-laser-radar-measurement-data-2.txt**

Total Measurements: 200

| Variable | RMSE |
|:--------:|:----:|
| px       | 0.172717 |
| py       | 0.178161 |
| vx       | 0.261516 | 
| vy       | 0.271265 |

---

## Future Plans

+ **Generate Additional Data** - use my own radar and lidar data collected using [utilities repo](https://github.com/udacity/CarND-Mercedes-SF-Utilities) Matlab scripts.
