![Banner](resources/loading_screen.png)

# Project 11 - Simulated Path Planning in C++

The goal of this project is to develop a path planner in C++ that can safely navigate around a virtual highway with other traffic that is driving at or below the 50 MPH speed limit. Specifically, the car must accomplish the following tasks:

+ **Speed Limit** - The car should try to go as close as possible to the 50 MPH speed limit, which means passing slower traffic when possible, note that other cars will try to change lanes too.

+ **Collisons** - The car should avoid hitting other cars at all cost as well as driving inside of the marked road lanes at all times, unless going from one lane to another.

+ **Distance Travelled** - The car should be able to make one complete loop around the 6946m highway. Since the car is trying to go 50 MPH, it should take a little over 5 minutes to complete 1 loop. 

+ **Acceleration & Jerk** - The car should not experience total acceleration over 10 m/s^2 and jerk that is greater than 50 m/s^3.

The car's localization and sensor fusion data is provided in addition to a sparse map list of waypoints around the highway. Each waypoint in the list contains [x, y, s, dx, dy] values. x and y are the waypoint's map coordinate position, the s value is the distance along the road to get to that waypoint in meters, the dx and dy values define the unit normal vector pointing outward of the highway loop. See [data/highway_map.csv](data/highway_map.csv) for more information.

The highway's waypoints are set up in an overlapping loop. Therefore, the frenet s value (the distance along the road) goes from 0 to 6945.554.

# Getting Started

+ The starter code for this project was taken from Udacity's ![repository](https://github.com/udacity/CarND-Path-Planning-Project). 

+ The Simulator used for this project can be downloaded [here] erm3 Simulator which contains the Path Planning Project from the [releases tab (https://github.com/udacity/self-driving-car-sim/releases).

### Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./path_planning`.

# Definitions

### Main Car's Localization Data (No Noise)

["x"] The car's x position in map coordinates

["y"] The car's y position in map coordinates

["s"] The car's s position in frenet coordinates

["d"] The car's d position in frenet coordinates

["yaw"] The car's yaw angle in the map

["speed"] The car's speed in MPH

### Previous Path Data

["previous_path_x"] The previous list of x points previously given to the simulator

["previous_path_y"] The previous list of y points previously given to the simulator

["end_path_s"] The previous list's last point's frenet s value

["end_path_d"] The previous list's last point's frenet d value

### Sensor Fusion Data

["sensor_fusion"] A 2d vector of cars and then that car's [car's unique ID, car's x position in map coordinates, car's y position in map coordinates, car's x velocity in m/s, car's y velocity in m/s, car's s position in frenet coordinates, car's d position in frenet coordinates. 

# Project Reflections

### Perfect Controller

To eliminate the need to implement a controller, the car uses a perfect controller and visits every (x,y) point it receives in the list every .02 seconds. The units for the (x,y) points are in meters and the spacing of the points determines the speed of the car. The vector going from a point to the next point in the list dictates the angle of the car. Acceleration both in the tangential and normal directions is measured along with the jerk, the rate of change of total Acceleration.

### Latency and Transitions

Some latency exists between the simulator running and the path planner returning a path; however, with optimized code this usually does not exceed 1 to 3 time steps. The simulator will continue using points that it was last given during a delay, so the transition is not always smooth. I chose to implement the spline library to combat this issue. After including spline.h in main.cpp, I use previous_path_x and previous_path_y to create new paths with smooth transitions provided by a spline.

# Dependencies

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
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `install-mac.sh` or `install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```

# General Notes

+ The code in this repository roughly follows [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html).
+ All of the code must be buildable with cmake and make.

# Future Plans

+ **MPC/PID Controller Implementation** - In the future, I would like to expand the functionality of this project to include either an MPC or a PID controller similar to the ones I built in previous projects.
