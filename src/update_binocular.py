import numpy as np
from scipy.linalg import expm
from tqdm import tqdm
import time

from utils import visualize_trajectory_2d,load_data

############## All Utility Functions ###############
def round_dot(vec):
    '''

    :param vec: A point in homogeneous co-ordinate
    :return: 0 representation for  the vector
    '''
    assert  vec.shape == (4,1)
    vec_homog = to_homog(vec)
    vec_hat = hat(vec_homog[0:3,0])
    vec_round_dot = np.hstack((np.eye(3),-vec_hat))
    vec_round_dot = np.vstack((vec_round_dot,np.zeros((1,6))))
    return vec_round_dot

def to_homog(vec):
    '''

    :param vec: A vector
    :return: A vector scaled to make sure the last component is one
    '''
    assert vec.shape == (4,1)
    return vec / vec[3,0]

def pi(point):
    '''
    :param : point in 3D
    :return: projected point in 2D
    '''
    point = point.reshape(4)
    return point / point[2]

def inv_pi(point):
    '''

    :param point: In 2D
    :return: in 3D - Gain depth info
    '''
    assert point.shape == (4,1)
    return point * point[3]

def deri_pi(point):
    '''

    :param point: derivative of the projection  this point is taken
    :return: derivative
    '''

    point = point.reshape(4)
    return np.array([[1,0,-point[0]/point[2],0],
                     [0,1,-point[1]/point[2],0],
                     [0,0,0,0],
                     [0,0,-point[3]/point[2],1]]) / point[2]
def hat(vec):
    '''
    This function computes the hat function
    '''
    assert vec.ndim == 1
    return np.array([[0,-vec[2],vec[1]],
                     [vec[2],0,vec[1]],
                     [-vec[1],vec[0],0]])

def curly_hat(omega_hat,v):
    '''
    This function Computes the curly hat operations
    '''
    v_hat = hat(v)
    curly_u = np.hstack((omega_hat,v_hat))
    curly_u = np.vstack((curly_u,np.hstack((np.zeros((3,3)),omega_hat))))

    return curly_u

############## All Utility Functions Ends ###############

def imu_ekf(data_set):
    '''
    This function performs the update for the data received from the
    Binocular Camera.
    This method is used to solve the predict only step in the problem
    :param data_set-> The data set which is to be read for processing
    :return -> The mean poses for the IMU is returned
    '''
    time_stamp,features,v,omega,K,b,cam_T_imu = load_data(data_set)
    start_time = time.time()

    z = features
    opt_T_imu = cam_T_imu
    #### Initializations ####
    prev_pose = np.eye(4)
    prev_cov = np.eye(6)
    pose_mean = np.zeros((4,4,time_stamp.shape[1]))
    for t in tqdm(range(time_stamp.shape[1]-1)):
        tau = time_stamp[0,t+1] - time_stamp[0,t]
        omega_hat = hat(omega[:,t])
        u_hat = np.hstack((omega_hat,v[:,t].reshape(3,1)))
        u_hat = np.vstack((u_hat,np.zeros((1,4))))

        #### Predict IMU Pose ####
        #### Mean ####
        pose_mean[:,:,t] = expm(-tau * u_hat) @ prev_pose
        prev_pose = pose_mean[:,:,t]
        #### Co-variance ####
        W = np.diag(np.random.randn(6))
        pose_cov = expm(-tau * curly_hat(omega_hat,v[:,t])) @ prev_cov \
                   @ expm(-tau * curly_hat(omega_hat,v[:,t])).T + W

    #visualize_trajectory_2d(pose_mean)
    print("Done IMU Predict and time taken is ", time.time()-start_time)
    return pose_mean


def slam_imu_predict(time_stamp,features,v,omega,K,b,cam_T_imu,t,prev_pose,prev_cov):
    '''
    This function performs the predict step for the
    Slam problem, This is called once for every time stamp
    Inputs: Along with the previous pose and covariance matrix
    Output: Predicted covaraince and mean for the IMU pose
    '''
    start_time = time.time()

    z = features
    opt_T_imu = cam_T_imu
    tau = time_stamp[0,t+1] - time_stamp[0,t]
    omega_hat = hat(omega[:,t])
    u_hat = np.hstack((omega_hat,v[:,t].reshape(3,1)))
    u_hat = np.vstack((u_hat,np.zeros((1,4))))

    #### Predict IMU Pose ####
    #### Mean ####
    pose_mean = expm(-tau * u_hat) @ prev_pose
    #### Co-variance ####
    W = np.diag(np.random.randn(6))
    pose_cov = expm(-tau * curly_hat(omega_hat,v[:,t])) @ prev_cov \
               @ expm(-tau * curly_hat(omega_hat,v[:,t])).T + W

    #visualize_trajectory_2d(pose_mean)
    #print("Done IMU Predict and time taken is ", time.time()-start_time)
    return pose_mean, pose_cov


def slam(data_set):
    '''
    This performs slam for the visual odometry data
    Step 1: Performs predict for IMU pose
    Step 2: Performs update for IMU pose and landmark points
            Substep:  Compute Jacobian for H_l and H_u
            Substep: Concatenate both of them
            Substep: Perform overall update for Covraince and Kalman Gain
            Substep: Perform individual update for the means of IMU pose and Landmark Locations
    :return: Plot of the localization and Mapping for the Particle
    '''
    time_stamp,z,v,omega,k,b,cam_T_imu = load_data(data_set)

    #Choosing Points in Map
    chose_landmark_option = 1
    if(chose_landmark_option == 1):
        chosen_landmarks = [i for i in range(z.shape[1]) if i%10 == 0]
    elif(chose_landmark_option == 2):
        chosen_landmarks = np.random.randint(0,z.shape[1],500)
    last_landmark = max(chosen_landmarks)

    #Temprory variables
    landmark_mean_cam = np.zeros(3)
    first_observation = np.zeros(3)


    #Projection Constants
    P_T = np.hstack((np.eye(3),np.zeros((3,1)))).T
    M = np.hstack((k[0:2,0:3],np.zeros((2,1))))
    M = np.vstack((M,M))

    landmark_mean = np.zeros((3 * len(chosen_landmarks))) # Total LandMarks are 3M
    state_cov = 2 * np.eye(3*len(chosen_landmarks)+6) #New State Variable with Size 3M+6

    imu_prev_pose, imu_prev_cov = np.eye(4),  np.eye(6) # To predict module Initialization
    pose_mean = np.zeros((4,4,features.shape[2])) #For plotting purpose size is 4x4xT
    for t in tqdm(range(features.shape[2]-1)):
        #### IMU Predict pos and covariance ####
        imu_pred_pos,imu_pred_cov = slam_imu_predict(time_stamp,z,v,omega,K,b,cam_T_imu,t,imu_prev_pose,imu_prev_cov)

        z_tik = np.zeros((4 * len(chosen_landmarks),1)) #Observation Model Readings
        z_observed = np.zeros((4 * len(chosen_landmarks),1)) #Sensor readings

        ### Find the legal Readings and Choose the one's in the Points of Interest ###
        z_sum = np.sum(z[:,0:last_landmark,t],axis=0)
        valid_scans = np.where(z_sum != -4)
        valid_and_relevant_scans = [scan for scan in valid_scans[0] if scan in chosen_landmarks]
        H_l = np.zeros((4*len(chosen_landmarks),3*len(chosen_landmarks)))
        H_u = np.zeros((4*len(chosen_landmarks),6))
        for scan in valid_and_relevant_scans:
            ###### Jacobian for Mapping Calculation #####
            scan_loc = chosen_landmarks.index(scan) # The location of the current scan in the original array
            str_4x,end_4x = scan_loc*4, scan_loc*4+4
            str_3x,end_3x = scan_loc*3, scan_loc*3+3

            ##### Initialization for scans seen for the first time ######
            if (np.all(landmark_mean[str_3x:end_3x] == first_observation)):
                ## Convert Z into Camera Cordinates
                landmark_mean_cam[2] = -M[2, 3] / (z[0, scan, t] - z[2, scan, t])
                landmark_mean_cam[1] = (z[1, scan, t] - M[1, 2]) * landmark_mean_cam[2] / M[1, 1]
                landmark_mean_cam[0] = (z[0, scan, t] - M[0, 2]) * landmark_mean_cam[2] / M[0, 0]
                landmark_mean_cam_homog = np.vstack((landmark_mean_cam.reshape(3, 1), 1))
                landmark_mean_homog = np.linalg.inv(cam_T_imu @ imu_pred_pos) @ landmark_mean_cam_homog
                landmark_mean[str_3x:end_3x] = landmark_mean_homog[0:3, 0]


            ##### Perform Update related Operations ######
            else:
                landmark_mean_homo = np.vstack((landmark_mean[str_3x:end_3x].reshape(3, 1), 1))
                landmark_camera = cam_T_imu @ imu_pred_pos @ landmark_mean_homo
                dpi_dq = deri_pi(landmark_camera)
                H_l[str_4x:end_4x,str_3x:end_3x] = M @  dpi_dq @ cam_T_imu @ imu_pred_pos @ P_T

                ###### Jacobian for IMU Calculation #####
                H_u[str_4x:end_4x,:] = M @ dpi_dq @ cam_T_imu @ round_dot(to_homog(imu_pred_pos @ landmark_mean_homo))
                ###### Observed vs Expected ######
                z_observed[str_4x:end_4x,0] = z[:,scan,t]
                z_tik[str_4x:end_4x,0] = M @ pi(landmark_camera)

        #### Update Combined Covariance####
        H = np.hstack((H_l,H_u)) #Main Jacobian
        N = np.diag(5 * np.random.rand(H.shape[0]))
        ###### If the inverse leads to Singularity Compute Another Noise ######
        try:
            Kalman_gain = state_cov @ H.T @ np.linalg.inv(H @ state_cov @ H.T + N)
        except:
            N = np.diag(6 * np.random.rand(H.shape[0]))
            Kalman_gain = state_cov @ H.T @ np.linalg.inv(H @ state_cov @ H.T + N)


        #### Update the Stat_covariance Matrix ####
        state_cov = (np.eye(3*len(chosen_landmarks)+6) - Kalman_gain @ H) @ state_cov
        ##IMU Mean Update##
        perturb_pos = Kalman_gain[-6:,:] @ (z_observed-z_tik) #Pick last few rows to get IMU details
        perturb_pos_hat = np.hstack((hat(perturb_pos[3:6,0]),perturb_pos[0:3,0].reshape(3,1)))
        perturb_pos_hat = np.vstack((perturb_pos_hat,np.zeros((1,4))))
        imu_update_pose = expm(perturb_pos_hat) @ imu_pred_pos
        pose_mean[:,:,t] = imu_update_pose

        ##LandMark Mean Update ##
        perturb_landmark = Kalman_gain[0:-6,:] @ (z_observed - z_tik) #Pick first 3M rows
        landmark_mean = landmark_mean + perturb_landmark.reshape(-1)

        #update imu pos  with the updated value of these varaibles
        imu_prev_pose = imu_update_pose

    visualize_trajectory_2d(pose_mean, landmark_mean.reshape(-1, 3).T)


def visual_ekf(pose_mean,z,k,b,cam_T_imu):
    '''
    :param pose_mean: The estimated pose for the IMU  Data set along with the Estimated pose of IMU
    Computes the Landmark update based on the assumption of IMU poses being golden
    Uses the Stereo Camera Model to get the output
    :return: Plot of the localization of the body along with the maps for the sourrounding
    '''

    print("Starting Mapping Update")
    start_time = time.time()
    num_landmark = z.shape[1]
    landmark_mean = np.zeros((3*num_landmark)) # 3M
    landmark_cov  = np.diag(1e-2*np.random.randn(3*num_landmark))
    landmark_mean_cam = np.zeros(3)
    landmark_mean_cam_homog = np.zeros((4,1))

    P_T = np.hstack((np.eye(3),np.zeros((3,1)))).T
    M = np.hstack((k[0:2,0:3],np.zeros((2,1))))
    M = np.vstack((M,M))
    M[2,3] = -k[0,0] * b #Disparity
    total_time = z.shape[2]

    no_observation = np.array([-1,-1,-1,-1])
    first_observation = np.zeros(3)
    for t in tqdm(range(total_time)):
        jacobian = np.zeros((4*num_landmark, 3*num_landmark))
        z_tik = np.zeros((4 * num_landmark))
        z_sum = np.sum(z[:,0:num_landmark,t],axis=0)
        valid_scans = np.where(z_sum != -4)
        #for landmark in range(num_landmark-1):
        for landmark in valid_scans[0]:
            lnd_mrk_strt, lnd_mrk_end = landmark * 3, landmark * 3 + 3
            if(np.all(landmark_mean[lnd_mrk_strt:lnd_mrk_end] == first_observation)):
                landmark_mean_cam[2] = -M[2,3] / (z[0,landmark,t] - z[2,landmark,t])
                landmark_mean_cam[1] = (z[1,landmark,t]  - M[1,2]) * landmark_mean_cam[2] / M[1,1]
                landmark_mean_cam[0] = (z[0,landmark,t]  - M[0,2]) * landmark_mean_cam[2] / M[0,0]
                landmark_mean_cam_homog = np.vstack((landmark_mean_cam.reshape(3,1),1))
                landmark_mean_homog = np.linalg.inv(cam_T_imu @ pose_mean[:,:,t]) @ landmark_mean_cam_homog
                landmark_mean[lnd_mrk_strt:lnd_mrk_end] = landmark_mean_homog[0:3,0]
                #initialize
            else:
                landmark_mean_homo = np.vstack((landmark_mean[lnd_mrk_strt:lnd_mrk_end].reshape(3,1),1))
                landmark_camera = cam_T_imu @ pose_mean[:, :, t] @ landmark_mean_homo
                dpi_dq = deri_pi(landmark_camera)
                strt,end = landmark*3,landmark*3 + 3 #Address
                z_tik = (M @ pi(landmark_camera)).flatten()
                jacobian = M @  dpi_dq @ cam_T_imu @ pose_mean[:,:,t] @ P_T
                k_gain = landmark_cov[strt:end,strt:end] @ jacobian.T @ \
                         np.linalg.inv(jacobian @  landmark_cov[strt:end,strt:end] @ jacobian.T \
                                       + np.diag(30 * np.random.randn(4))) #np.diag(1e2) also worked
                landmark_mean[strt:end] = landmark_mean[strt:end] + k_gain @ (z[:,landmark,t] - z_tik)
                landmark_cov[strt:end,strt:end] = (np.eye(3) - k_gain @ jacobian) @ landmark_cov[strt:end,strt:end]

    print("Done Mapping update and time taken is ", time.time()-start_time)
    visualize_trajectory_2d(pose_mean,landmark_mean.reshape(-1,3).T)


if __name__ == '__main__':
    dataset_list = ['data/0022.npz','data/0027.npz','data/0034.npz']

    for data_set in dataset_list:
        t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(data_set)
        ### Run Part a and Part b ###
        visual_ekf(imu_ekf(data_set),features,K,b,cam_T_imu)

        ### Run Part c ###
        slam(data_set)


