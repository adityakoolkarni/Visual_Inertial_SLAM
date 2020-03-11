import numpy as np
from scipy.linalg import expm

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

def imu_ekf():
    '''
    This function performs the update for the data received from the
    Binocular Camera
    '''
    time_stamp,features,v,omega,K,b,cam_T_imu = load_data('data/0022.npz')

    z = features
    opt_T_imu = cam_T_imu
    prev_pose = np.eye(4)
    prev_cov = np.eye(6)
    pose_mean = np.zeros((4,4,time_stamp.shape[1]))
    W = 1e-3 * prev_cov
    for t in range(time_stamp.shape[1]-1):
        tau = time_stamp[0,t+1] - time_stamp[0,t]
        omega_hat = hat(omega[:,t])
        u_hat = np.hstack((omega_hat,v[:,t].reshape(3,1)))
        u_hat = np.vstack((u_hat,np.zeros((1,4))))

        #### Predict IMU Pose ####
        #### Mean ####
        pose_mean[:,:,t] = expm(-tau * u_hat) @ prev_pose
        prev_pose = pose_mean[:,:,t]
        #### Co-variance ####
        pose_cov = expm(-tau * curly_hat(omega_hat,v[:,t])) @ prev_cov \
                   @ expm(-tau * curly_hat(omega_hat,v[:,t])).T + W




    #visualize_trajectory_2d(pose_mean)
    return pose_mean


def pi(point):
    '''
    :param : point in 3D
    :return: projected point in 2D
    '''
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

    return np.array([[1,0,-point[0]/point[2],0],
                     [0,1,-point[1]/point[2],0],
                     [0,0,1,0],
                     [0,0,-point[3]/point[2],1]]) / point[2]

def visual_ekf(pose_mean,z,k,b,cam_T_imu):
    '''
    :param pose_mean: The estimated pose for the IMU
    :return:
    '''

    num_landmark = 2000#z.shape[1]
    landmark_mean = np.zeros((3*num_landmark)) # 3M
    #landmark_mean = np.zeros((3,num_landmark)) # 3,M
    landmark_cov  = np.zeros((3*num_landmark,3*num_landmark)) #3M x 3M

    P_T = np.hstack((np.eye(3),np.zeros((3,1)))).T
    M = np.hstack((k[0:2,0:3],np.zeros((2,1))))
    M = np.vstack((M,M))
    M[2,3] = -k[0,0] * b #Disparity

    total_time = z.shape[2]
    no_observation = np.array([-1,-1,-1,-1])
    first_observation = np.zeros(3)
    for t in range(total_time):
        jacobian = np.zeros((4*num_landmark, 3*num_landmark))
        z_tik = np.zeros((3 * num_landmark))
        for landmark in range(num_landmark-1):
            lnd_mrk_strt, lnd_mrk_end = landmark * 3, landmark * 3 + 3
            if(np.all(z[:,landmark,t] == no_observation)):
                pass
            elif(np.all(landmark_mean[lnd_mrk_strt:lnd_mrk_end] == first_observation)):
                landmark_mean[lnd_mrk_strt:lnd_mrk_end] = inv_pi(np.linalg.inv(M.T @ M) @ M.T \
                                                              @ z[:,landmark,t].reshape(4,1))[0:3,0]
                #initialize
            else:
                landmark_mean_homo = np.vstack((landmark_mean[lnd_mrk_strt:lnd_mrk_end].reshape(3,1),1))
                landmark_camera = pi(cam_T_imu @ pose_mean[:, :, t] @ landmark_mean_homo)
                dpi_dq = deri_pi(landmark_camera)
                jcbn_rstart,jcbn_rend = landmark*4,landmark*4 + 4 #Address
                jcbn_cstart,jcbn_cend = landmark*3,landmark*3 + 3 #Address
                z_tik[lnd_mrk_strt:lnd_mrk_end] = M @ landmark_camera
                jacobian[jcbn_rstart:jcbn_rend,jcbn_cstart:jcbn_cend] = M @  dpi_dq @ cam_T_imu @ pose_mean[:,:,t] @ P_T

        # update with Jacobian
        k_gain = landmark_cov @ jacobian.T @ (jacobian @ landmark_cov @ jacobian.T) # TODO Noise
        landmark_mean = landmark_mean + k_gain @ (z[:,:,t].T.reshape(-1,1) - z_tik)

    visualize_trajectory_2d(pose_mean,landmark_mean.reshape(-1,3).T)



if __name__ == '__main__':
    t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data('data/0022.npz')
    print('t',t.shape)
    print('m',features.shape)
    print('M',K.shape)
    print('M',K)
    print('b',b.shape)
    visual_ekf(imu_ekf(),features,K,b,cam_T_imu)


