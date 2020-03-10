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


    visualize_trajectory_2d(pose_mean)


if __name__ == '__main__':
    t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data('data/0022.npz')
    print('t',t.shape)
    print('m',features.shape)
    print('M',K.shape)
    print('M',K)
    print('b',b.shape)
    imu_ekf()


