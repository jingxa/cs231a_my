import sys
import numpy as np
import os
import scipy.misc
from scipy.optimize import least_squares
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sfm_utils import *

'''
ESTIMATE_INITIAL_RT from the Essential Matrix, we can compute 4 initial
guesses of the relative RT between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
Returns:
    RT: A 4x3x4 tensor in which the 3x4 matrix RT[i,:,:] is one of the
        four possible transformations
'''
def estimate_initial_RT(E):
    # SVD分解
    u, s, vt = np.linalg.svd(E)

    # 计算R
    z = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 0]
    ])

    w = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])

    M = u.dot(z).dot(u.T)
    Q1 = u.dot(w).dot(vt)
    R1 = np.linalg.det(Q1) * Q1

    Q2 = u.dot(w.T).dot(vt)
    R2 = np.linalg.det(Q2) * Q2

    # 计算 T,T为u的第三列
    T1 = u[:, 2].reshape(-1,1)
    T2 = - T1

    # R 有两种， T有两种，一共四中可能
    R_set = [R1, R2]
    T_set = [T1, T2]
    RT_set = []
    for i in range(len(R_set)):
        for j in range(len(T_set)):
            RT_set.append(np.hstack( (R_set[i], T_set[j]) ))

    RT = np.zeros((4, 3, 4))
    for i in range(RT.shape[0]):
        RT[i, :, :] = RT_set[i]


    return RT

'''
LINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point is the best linear estimate
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def linear_estimate_3d_point(image_points, camera_matrices):
    # 传入两个点， 两个相机矩阵
    N = image_points.shape[0]   # N == 2
    # 建立系数矩阵
    A = np.zeros((2*N, 4))  # 每一点建立两个等式
    A_set = []

    for i in range(N):
        pi = image_points[i]
        Mi = camera_matrices[i]
        x = pi[0] * Mi[2] - Mi[0]
        y = pi[1] * Mi[2] - Mi[1]
        A_set.append(x)
        A_set.append(y)


    for i in range(A.shape[0]):
        A[i] = A_set[i]

    u, s, vt = np.linalg.svd(A)
    p = vt[-1]      # 取最后一列
    p = p / p[-1]       # 齐次坐标，最后一个为1
    p = p[:3]       # 取前三个，3D坐标

    return p


'''
REPROJECTION_ERROR given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    error - the 2Mx1 reprojection error vector
'''
def reprojection_error(point_3d, image_points, camera_matrices):
    # 图像点的数量    3d点：1， 2d点： 2， 相机矩阵 2
    N = image_points.shape[0]
    # 建立其次坐标系
    point_3d_homo = np.hstack((point_3d, 1))

    error_set = []
    # 计算误差
    for i in range(N):
        pi = image_points[i]
        Mi = camera_matrices[i]
        Yi = Mi.dot(point_3d_homo)  # 3d点的投影

        pi_prime = 1.0 / Yi[2] * np.array([Yi[0], Yi[1]])       # 转变为2d点
        error_i = pi_prime - pi
        error_set.append(error_i[0])        # x 误差
        error_set.append(error_i[1])        # y 误差

    # 变换成 array
    error = np.array(error_set)
    return error


'''
JACOBIAN given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    jacobian - the 2Mx3 Jacobian matrix
'''
def jacobian(point_3d, camera_matrices):
    # 一个 3D点， 2个相机矩阵
    K = camera_matrices.shape[0]

    J = np.zeros((2 * K, 3))        # 2*k, 3

    #  齐次坐标系
    point_3d_homo = np.hstack((point_3d, 1))

    J_tmp = []
    # 计算 雅克比矩阵
    for i in range(K):
        Mi = camera_matrices[i]
        pi = Mi.dot(point_3d_homo)      # 投影点
        Jix =(pi[2]*Mi[0] - pi[0] * Mi[2] ) / pi[2]**2      # x
        Jiy =(pi[2]*Mi[1] - pi[1] * Mi[2]) / pi[2]**2       # y
        J_tmp.append(Jix)
        J_tmp.append(Jiy)

    for i in range(J.shape[0]):
        J[i] = J_tmp[i][:3]         # J_tmp 每一行包含齐次坐标，去掉最后一列
    return J



'''
NONLINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point that iteratively updates the points
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def nonlinear_estimate_3d_point(image_points, camera_matrices):
    # 计算初始估计的真实点
    P = linear_estimate_3d_point(image_points, camera_matrices)

    # 迭代10 次
    for i in range(10):
        e = reprojection_error(P, image_points, camera_matrices)
        J = jacobian(P, camera_matrices)
        P -= np.linalg.inv(J.T.dot(J)).dot(J.T).dot(e)

    return P

'''
ESTIMATE_RT_FROM_E from the Essential Matrix, we can compute  the relative RT 
between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
    image_points - N measured points in each of the M images (NxMx2 matrix)
    K - the intrinsic camera matrix
Returns:
    RT: The 3x4 matrix which gives the rotation and translation between the 
        two cameras
'''
def estimate_RT_from_E(E, image_points, K):
    # 获得四队RT
    RT = estimate_initial_RT(E)
    count = np.zeros((1, 4))        # 每一对的估计得分
    I0 = np.array([[1.0, 0, 0, 0],
                   [0, 1.0, 0, 0],
                   [0, 0, 1.0, 0]])     # [I 0] 设第一个相机为初始位置

    M1 = K.dot(I0)      # 第一个相机的变换矩阵

    camera_matrices = np.zeros((2, 3, 4))       # 建立两个相机变换矩阵
    camera_matrices[0] = M1


    for i in range(RT.shape[0]):        # 对每一个RT进行测试
        rt = RT[i]
        M2_i = K.dot(rt)
        camera_matrices[1] = M2_i       # 临时第二个相机变换矩阵
        for j in range(image_points.shape[0]):      # 估计 3D点
            point_j_3d = nonlinear_estimate_3d_point(image_points[j], camera_matrices)
            pj_homo = np.vstack((point_j_3d.reshape(3, 1), [1]))      # 转换为齐次坐标
            pj_prime = camera1tocamera2(pj_homo, rt)      # 将3d点变换到相对的相机坐标系下
            if pj_homo[2] > 0 and pj_prime[2] > 0:
                count[0, i] += 1     # 3D点面向相机 ，z值为正， 计算正确


    maxIdx = np.argmax(count)
    maxRT = RT[maxIdx]
    return maxRT


def camera1tocamera2(P, RT):
    P_homo = np.array([P[0], P[1], P[2], 1.0])
    A = np.zeros((4, 4))
    A[0:3, :] = RT
    A[3, :] = np.array([0, 0, 0, 1.0])

    P_prim_homo = A.dot(P_homo.T)
    P_prim_homo /= P_prim_homo[3]
    P_prime = P_prim_homo[0:3]

    return P_prime




if __name__ == '__main__':
    run_pipeline = True

    # Load the data
    image_data_dir = 'data/statue/'
    unit_test_camera_matrix = np.load('data/unit_test_camera_matrix.npy')
    unit_test_image_matches = np.load('data/unit_test_image_matches.npy')
    image_paths = [os.path.join(image_data_dir, 'images', x) for x in
        sorted(os.listdir('data/statue/images')) if '.jpg' in x]
    focal_length = 719.5459
    matches_subset = np.load(os.path.join(image_data_dir,
        'matches_subset.npy'), encoding='latin1')[0, :]
    dense_matches = np.load(os.path.join(image_data_dir, 'dense_matches.npy'), encoding='latin1')
    fundamental_matrices = np.load(os.path.join(image_data_dir,
        'fundamental_matrices.npy'), encoding='latin1')[0, :]

    # Part A: Computing the 4 initial R,T transformations from Essential Matrix
    print ('-' * 80)
    print ("Part A: Check your matrices against the example R,T")
    print ('-' * 80)
    K = np.eye(3)
    K[0,0] = K[1,1] = focal_length
    E = K.T.dot(fundamental_matrices[0]).dot(K)
    im0 = scipy.misc.imread(image_paths[0])
    im_height, im_width, _ = im0.shape
    example_RT = np.array([[0.9736, -0.0988, -0.2056, 0.9994],
        [0.1019, 0.9948, 0.0045, -0.0089],
        [0.2041, -0.0254, 0.9786, 0.0331]])
    print ("Example RT:\n", example_RT)
    estimated_RT = estimate_initial_RT(E)
    print ("Estimated RT:\n", estimated_RT)




    # # Part B: Determining the best linear estimate of a 3D point
    print ('-' * 80)
    print ('Part B: Check that the difference from expected point ')
    print ('is near zero')
    print ('-' * 80)
    camera_matrices = np.zeros((2, 3, 4))
    camera_matrices[0, :, :] = K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))
    camera_matrices[1, :, :] = K.dot(example_RT)
    unit_test_matches = matches_subset[0][:,0].reshape(2,2)
    estimated_3d_point = linear_estimate_3d_point(unit_test_matches.copy(),
        camera_matrices.copy())
    expected_3d_point = np.array([0.6774, -1.1029, 4.6621])
    print ("Difference: ", np.fabs(estimated_3d_point - expected_3d_point).sum())




    # Part C: Calculating the reprojection error and its Jacobian
    print ('-' * 80)
    print ('Part C: Check that the difference from expected error/Jacobian ')
    print ('is near zero')
    print ('-' * 80)
    estimated_error = reprojection_error(
            expected_3d_point, unit_test_matches, camera_matrices)
    estimated_jacobian = jacobian(expected_3d_point, camera_matrices)
    expected_error = np.array((-0.0095458, -0.5171407,  0.0059307,  0.501631))
    print ("Error Difference: ", np.fabs(estimated_error - expected_error).sum())
    expected_jacobian = np.array([[ 154.33943931, 0., -22.42541691],
         [0., 154.33943931, 36.51165089],
         [141.87950588, -14.27738422, -56.20341644],
         [21.9792766, 149.50628901, 32.23425643]])
    print ("Jacobian Difference: ", np.fabs(estimated_jacobian
        - expected_jacobian).sum())







    # Part D: Determining the best nonlinear estimate of a 3D point
    print ('-' * 80)
    print ('Part D: Check that the reprojection error from nonlinear method')
    print ('is lower than linear method')
    print ('-' * 80)
    estimated_3d_point_linear = linear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    estimated_3d_point_nonlinear = nonlinear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    error_linear = reprojection_error(
        estimated_3d_point_linear, unit_test_image_matches,
        unit_test_camera_matrix)
    print ("Linear method error:", np.linalg.norm(error_linear))
    error_nonlinear = reprojection_error(
        estimated_3d_point_nonlinear, unit_test_image_matches,
        unit_test_camera_matrix)
    print ("Nonlinear method error:", np.linalg.norm(error_nonlinear))





    # Part E: Determining the correct R, T from Essential Matrix
    print ('-' * 80)
    print ("Part E: Check your matrix against the example R,T")
    print ('-' * 80)
    estimated_RT = estimate_RT_from_E(E,
        np.expand_dims(unit_test_image_matches[:2,:], axis=0), K)
    print ("Example RT:\n", example_RT)
    print
    print ("Estimated RT:\n", estimated_RT)

    # Part F: Run the entire Structure from Motion pipeline
    if not run_pipeline:
        sys.exit()
    print ('-' * 80)
    print ('Part F: Run the entire SFM pipeline')
    print ('-' * 80)
    frames = [0] * (len(image_paths) - 1)
    for i in range(len(image_paths)-1):
        frames[i] = Frame(matches_subset[i].T, focal_length,
                fundamental_matrices[i], im_width, im_height)
        bundle_adjustment(frames[i])
    merged_frame = merge_all_frames(frames)

    # Construct the dense matching
    camera_matrices = np.zeros((2,3,4))
    dense_structure = np.zeros((0,3))
    for i in range(len(frames)-1):
        matches = dense_matches[i]
        camera_matrices[0,:,:] = merged_frame.K.dot(
            merged_frame.motion[i,:,:])
        camera_matrices[1,:,:] = merged_frame.K.dot(
                merged_frame.motion[i+1,:,:])
        points_3d = np.zeros((matches.shape[1], 3))
        use_point = np.array([True]*matches.shape[1])
        for j in range(matches.shape[1]):
            points_3d[j,:] = nonlinear_estimate_3d_point(
                matches[:,j].reshape((2,2)), camera_matrices)
        dense_structure = np.vstack((dense_structure, points_3d[use_point,:]))

    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    ax.scatter(dense_structure[:,0], dense_structure[:,1], dense_structure[:,2],
        c='k', depthshade=True, s=2)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(0, 10)
    ax.view_init(-100, 90)

    plt.show()
