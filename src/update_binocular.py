import numpy as np
from scipy.linalg import expm
from tqdm import tqdm
import time

from utils import visualize_trajectory_2d,load_data


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

def imu_ekf(data_set):
    '''
    This function performs the update for the data received from the
    Binocular Camera
    '''
    time_stamp,features,v,omega,K,b,cam_T_imu = load_data(data_set)
    start_time = time.time()

    z = features
    opt_T_imu = cam_T_imu
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
        W = np.random.randn(1) * prev_cov
        pose_cov = expm(-tau * curly_hat(omega_hat,v[:,t])) @ prev_cov \
                   @ expm(-tau * curly_hat(omega_hat,v[:,t])).T + W

    #visualize_trajectory_2d(pose_mean)
    print("Done IMU Predict and time taken is ", time.time()-start_time)
    return pose_mean


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
    :return: in 3D
    '''
    assert point.shape == (4,1)
    return point * point[3]

def deri_pi(point):
    '''

    :param point: derivative at this point is taken
    :return: derivative
    '''

    point = point.reshape(4)
    return np.array([[1,0,-point[0]/point[2],0],
                     [0,1,-point[1]/point[2],0],
                     [0,0,0,0],
                     [0,0,-point[3]/point[2],1]]) / point[2]

def visual_ekf(pose_mean,z,k,b,cam_T_imu):
    '''
    :param pose_mean: The estimated pose for the IMU
    :return:
    '''

    print("Starting Mapping Update")
    start_time = time.time()
    num_landmark = z.shape[1]
    landmark_mean = np.zeros((3*num_landmark)) # 3M
    #landmark_mean = np.zeros((3,num_landmark)) # 3,M
    #landmark_cov  = 1e-3 * np.eye(3*num_landmark) #3M x 3M
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
                #landmark_mean[lnd_mrk_strt:lnd_mrk_end] = inv_pi(np.linalg.inv(M.T @ M) @ M.T \
                #                                              @ z[:,landmark,t].reshape(4,1))[0:3,0]
                ## Convert Z into Camera Cordinates
                landmark_mean_cam[2] = -M[2,3] / (z[0,landmark,t] - z[2,landmark,t])
                landmark_mean_cam[1] = (z[1,landmark,t]  - M[1,2]) * landmark_mean_cam[2] / M[1,1]
                landmark_mean_cam[0] = (z[0,landmark,t]  - M[0,2]) * landmark_mean_cam[2] / M[0,0]
                landmark_mean_cam_homog = np.vstack((landmark_mean_cam.reshape(3,1),1))
                landmark_mean_homog = np.linalg.inv(cam_T_imu @ pose_mean[:,:,t]) @ landmark_mean_cam_homog
                landmark_mean[lnd_mrk_strt:lnd_mrk_end] = landmark_mean_homog[0:3,0]
                #initialize
            else:
                landmark_mean_homo = np.vstack((landmark_mean[lnd_mrk_strt:lnd_mrk_end].reshape(3,1),1))
                landmark_camera = pi(cam_T_imu @ pose_mean[:, :, t] @ landmark_mean_homo)
                dpi_dq = deri_pi(landmark_camera)
                strt,end = landmark*3,landmark*3 + 3 #Address
                z_tik = (M @ landmark_camera).flatten()
                jacobian = M @  dpi_dq @ cam_T_imu @ pose_mean[:,:,t] @ P_T
                k_gain = landmark_cov[strt:end,strt:end] @ jacobian.T @ \
                         np.linalg.inv(jacobian @  landmark_cov[strt:end,strt:end] @ jacobian.T \
                                       + np.diag(30 * np.random.randn(4))) #np.diag(1e2) also worked
                landmark_mean[strt:end] = landmark_mean[strt:end] + k_gain @ (z[:,landmark,t] - z_tik)
                landmark_cov[strt:end,strt:end] = (np.eye(3) - k_gain @ jacobian) @ landmark_cov[strt:end,strt:end]



        # update with Jacobian
        #TODO: This may not be required moved to loop above
        #start_time = time.time()
        #i_x_v = np.kron(np.eye(num_landmark), np.random.randn(1)* np.eye(4))
        #print(time.time()-start_time)
        #start_time = time.time()
        #k_gain = landmark_cov @ jacobian.T @ np.linalg.inv(jacobian @ landmark_cov @ jacobian.T + i_x_v) # TODO Noise
        #print(time.time()-start_time)
        #start_time = time.time()
        #landmark_mean = landmark_mean + k_gain @ (z[:,0:num_landmark,t].flatten()- z_tik)
        #print(time.time()-start_time)

    print("Done Mapping update and time taken is ", time.time()-start_time)
    visualize_trajectory_2d(pose_mean,landmark_mean.reshape(-1,3).T)



if __name__ == '__main__':
    dataset_list = ['data/0022.npz','data/0027.npz','data/0034.npz']

    for data_set in dataset_list:
        t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(data_set)
        #print('t',t.shape)
        #print('m',features.shape)
        #print('M',K.shape)
        #print('M',K)
        visual_ekf(imu_ekf(data_set),features,K,b,cam_T_imu)


