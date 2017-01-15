![ScreenShot](images/loading_screen.png)
# Udacity's Self-Driving Car Nanodegree Program
## Project 3 - Behavioral Cloning
### _How to Train a Deep Neural Network to Drive a Simulated Car_ 
This project builds and implements a behavioral cloning (end-to-end) network that drives a simulated autonomous vehicle around two different test tracks. The first track is used to generate data that the car will use to react and recover in changing road conditions. Data was collected in **"TRAINING MODE"** on the simulator for several laps and saved to **driving_log.csv**. The data was prepared and used to train the deep neural network model as outlined below.

![ScreenShot](images/main_menu.png)

**"AUTONOMOUS MODE"** is used to test the model's performance and ensure that it is not overfitting the data after training. Driving simulation is accomplished by running the car's drive server using the following code from the Terminal after a cd into the project's directory:

#### $ python drive.py model.json


![ScreenShot](images/first_track.png)

The following naming conventions have been used in this project:

+ **model.py** - The script used to create and train the model. SDCND_P3_Hiddink.ipynb serves as the de facto model.py script in this implementation. This allowed the use of Jupyter notebooks and made up-front code writing more straightforward.

+ **drive.py** - The script to drive the car. The original file was kept largely the same, with the exception of a crop function. The script is in the home directory as drive.py.

+ **model.json** - The model architecture. The **archive** folder contains the historical versions of the model, model.json, that were created throughout this project's development.

+ **model.h5** - The model weights. The **archive** folder contains the historical versions of the model's weights, model.h5, that were created throughout this project's development.

+ **README.md** - explains the structure of your network and training approach. This is the **README.md** file.

### Model Architecture
The table below shows the model architecture used in this project. This CNN architecture has been successfully implemented in other steering angle solutions such as comma.ai (https://github.com/commaai/research), NVIDIA (https://arxiv.org/pdf/1604.07316v1.pdf), and in other student repos, such as ksakmann (https://github.com/ksakmann/CarND-BehavioralCloning/) and diyjac's repo on GitHub (https://github.com/diyjac/SDC-P3). 
![ScreenShot](images/model_architecture.png
Although slightly different in their approaches, each of the models mentioned above have the same thing in common: the data that they were supplied was heavily augmented in several ways in order to increase the model's exposure to changing road conditions. This project draws heavily on these techniques, and attempts to combine them for increased performance.

The model is built using the Keras library and is a Convolutional Neural Network (CNN) with four convolutional layers. The sizes of each layer were chosen to decrease the kernel size by a factor of 2 after each convolution. ReLu activations are used throughout the model, and two dropout layers help to reduce the tendancy toward overfitting.

### Training the Model (model.py)

This is the second track available in the simulator. Once the model is trained, this track used to verify performance on new terrain.

![ScreenShot](images/second_track.png)

#### Running the Model in the Simulator

!open -a Behavioral_Cloning_Simulator

#### Run the drive server to begin autonomous steering
!python drive.py model.json

#### References

#### Further plans
