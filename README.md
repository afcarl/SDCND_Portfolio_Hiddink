## Udacity's Self-Driving Car Nanodegree Program
#### Project 3 - Behavioral Cloning

![ScreenShot](images/loading_screen.png)

This project builds and implements a behavioral cloning (end-to-end) network that drives a simulated autonomous vehicle around two different test tracks. The first track is used to generate data that the car will use to react and recover in changing road conditions. Data is recorded in "TRAINING MODE" for several laps before being saved to the Jupyter notebook heirarchy in the data folder. From there, the data is preprocessed and used to train the pipeline's deep neural network model, as explained below.

![ScreenShot](images/main_menu.png)

**"AUTONOMOUS MODE"** is used to test the model's performance and ensure that it is not overfitting the data after training. Driving simulation is accomplished by running the car's drive server using the following code from the Terminal after a cd into the project's directory:

#### $ python drive.py model.json

Selecting **"AUTONOMOUS MODE"** will start running the model and the car will begin driving on its own.

![ScreenShot](images/first_track.png)

Udacity has requested that the following conventions be used for this project:

+ **model.py** - The script used to create and train the model.

+ **drive.py** - The script to drive the car. You can feel free to resubmit the original drive.py or make modifications and submit your modified version.

+ **model.json** - The model architecture.

+ **model.h5** - The model weights.

+ **README.md** - explains the structure of your network and training approach. While we recommend using English for good practice, writing in any language is acceptable (reviewers will translate). There is no minimum word count so long as there are complete descriptions of the problems and the strategies. See the rubric for more details about the expectations.

**Because the model is being run in a Jupyter notebook, model.py is the model_py function in the pipeline rather than a separate file. Additionally, README.md is an overall summary that references this pipeline for further detail.**

#### Model Architecture



#### Training the Model (model.py)


This is the second track available in the simulator. Once the model is trained, this track used to verify performance on new terrain.

![ScreenShot](images/second_track.png)

#### Running the Model in the Simulator

!open -a Behavioral_Cloning_Simulator

#### Run the drive server to begin autonomous steering
!python drive.py model.json


#### References

#### Further plans
