[//]: # (Image References)
[atom]: https://github.com/nhiddink/CarND_P6_Extended_Kalman_Filters/blob/master/screenshots/atom.png
[loading_screen]: https://github.com/nhiddink/CarND_P5_Vehicle_Detection_and_Tracking/blob/master/resources/screenshots/loading_screen.png
[process_flow]: https://github.com/nhiddink/CarND_P6_Extended_Kalman_Filters/blob/master/screenshots/process_flow.png
[sensor_pros_cons]: https://github.com/nhiddink/CarND_P6_Extended_Kalman_Filters/blob/master/screenshots/sensor_pros_cons.png

[![loading_screen](https://github.com/nhiddink/CarND_P5_Vehicle_Detection_and_Tracking/blob/master/resources/screenshots/loading_screen.png)](http://www.udacity.com/drive)

## Udacity's Self-Driving Car Nanodegree Program
### Project 6 - Extended Kalman Filters

---

The starter code and input data for this project was taken from the following repository on GitHub: 
https://github.com/udacity/CarND-Extended-Kalman-Filter-Project

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

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./ExtendedKF path/to/input.txt path/to/output.txt`. You can find
   some sample inputs in 'data/'.
    - eg. `./ExtendedKF ../data/sample-laser-radar-measurement-data-1.txt output.txt`

---

## Code Style

In general, this repository conforms to [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html).

---

## Overview

This project is an implementation of an Extended Kalman Filter in C++. The filter utilizes both laser and radar data inputs, as each has its own advantages. Laser data can accurately define an object's position in space. Radar data provides valuable speed data that can be used to track an object's velocity relative to the sensor. The following schematic outlines these concepts:

![sensor_pros_cons]

Combining data from multiple sensors such that the resulting information has less uncertainty than either of the originals alone is the essence of [Sensor Fusion](https://en.wikipedia.org/wiki/Sensor_fusion).

The process flow of the project is described below. Each measurement type is handled separately because they are recorded in different coordinate systems. 

![process_flow]

---

## Results (Terminal)

The following results for Root Mean Square Error (RMSE) Accuracy were produced for each set of data, respectively:

**sample-laser-radar-measurement-data-1.txt**

| Variable | RMSE |
|:--------:|:----:|
| px       | 0.0651649 |
| py       | 0.0605378 |
| vx       | 0.54319   | 
| vy       | 0.544191  |

**sample-laser-radar-measurement-data-2.txt**

| Variable | RMSE |
|:--------:|:----:|
| px       | 0.208971 |
| py       | 0.214995 |
| vx       | 0.510136 | 
| vy       | 0.808932 |

---

## Future Plans

+ **Generate Additional Data** - use my own radar and lidar data collected using [utilities repo](https://github.com/udacity/CarND-Mercedes-SF-Utilities) Matlab scripts.
