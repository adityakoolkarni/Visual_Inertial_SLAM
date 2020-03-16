This code is written in partial fullfillment for the final project of ECE 276A. THis project aims to solve the problem of visual-inertial SLAM using stereo setup along with the IMU. The code is arranged as follows

The src folder contains all the files required to run Visual-Intertial SLAM. The description of files contained is
1. utils.py - Contains many helped functions that help to read the SLAM data and also help to visualize SLAM
2. update_binocular.py - This is the heart of SLAM and has all methods required to Perform. It has functions to perform IMU predict only, Update only, Landmark Update o 			only steps. Additionally it can merge all these steps into one step called slam which can perform all these activitiees together to perform and 			visualize SLAM

To run the part a,b and c use : python update_binocular.py on command line. First part a runs followed by b and c. 

Requirements -> Numpy, Python 3.7, Scipy, Matplotlib only

####### Thank You #########
Aditya Kulkarni
UC San Diego
