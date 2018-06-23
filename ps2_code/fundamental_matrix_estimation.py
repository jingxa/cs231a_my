import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import scipy.io as sio
from epipolar_utils import *

'''
LLS_EIGHT_POINT_ALG  computes the fundamental matrix from matching points using 
linear least squares eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the linear least squares eight
point algorithm works
'''

def lls_eight_point_alg(points1, points2):
    len = points1.shape[0]

    W = np.zeros((len, 9))   # 37 * 9 齐次矩阵
    for i in range(len):
        u1 = points1[i, 0]
        v1 = points1[i, 1]
        u2 = points2[i, 0]
        v2 = points2[i, 1]
        W[i] = np.r_[u1*u2, u2*v1, u2, v2*u1, v1*v2, v2, u1, v1, 1]

    # SVD
    U, S, VT = np.linalg.svd(W, full_matrices=True)
    f = VT[-1, :]   # 最后一行为最优解
    F_hat = np.reshape(f, (3, 3))    # 最小二乘的近似

    # 计算rank =2 的F
    U, S_hat, VT = np.linalg.svd(F_hat, full_matrices=True)
    s = np.zeros((3, 3))   # sigma 矩阵
    s[0, 0] = S_hat[0]
    s[1, 1] = S_hat[1]  # sigma 的Rank为2
    F = np.dot(U, np.dot(s, VT))

    return F







'''
NORMALIZED_EIGHT_POINT_ALG  computes the fundamental matrix from matching points
using the normalized eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the normalized eight
point algorithm works
'''
def normalized_eight_point_alg(points1, points2):
    N = points1.shape[0]
    points1_uv = points1[:, 0:2]
    points2_uv = points2[:, 0:2]    # 取x,y 坐标
    #
    # 取坐标均值
    points1_mean = np.mean(points1_uv, axis=0)
    points2_mean = np.mean(points2_uv, axis=0)

    # 点集的到中心的差
    points1_new = points1_uv - points1_mean
    points2_new = points2_uv - points2_mean

    # 计算缩放参数
    scale = np.sqrt(np.sum(points1_new**2)/N)
    scale1 = np.sqrt(2 / (np.sum(points1_new**2)/N * 1.0))
    scale2 = np.sqrt(2 / (np.sum(points2_new**2)/N * 1.0))

    # 归一化矩阵
    T1 = np.array([
        [scale1, 0, -points1_mean[0] * scale1],
        [0, scale1, -points1_mean[1] * scale2],
        [0, 0, 1]
    ])

    T2 = np.array([
        [scale2, 0, -points1_mean[0] * scale1],
        [0, scale2, -points1_mean[1] * scale2],
        [0, 0, 1]
    ])

    # 对坐标点变换
    q1 = T1.dot(points1.T).T    # N * 3
    q2 = T2.dot(points2.T).T    # N * 3

    # 八点算法
    Fq = lls_eight_point_alg(q1, q2)

    #反归一化
    F = T2.T.dot(Fq).dot(T1)

    return F








'''
PLOT_EPIPOLAR_LINES_ON_IMAGES given a pair of images and corresponding points,
draws the epipolar lines on the images
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    im1 - a HxW(xC) matrix that contains pixel values from the first image 
    im2 - a HxW(xC) matrix that contains pixel values from the second image 
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    Nothing; instead, plots the two images with the matching points and
    their corresponding epipolar lines. See Figure 1 within the problem set
    handout for an example
'''
def plot_epipolar_lines_on_images(points1, points2, im1, im2, F):
    plt.subplot(1, 2, 1)  # 建立1*2 的图
    line1 = F.T.dot(points2.T)   # p2到p1面上的极线     3 * N
    N1 = line1.shape[1]     # 极线的数量
    for i in range(N1):
        A = line1[0, i]
        B = line1[1, i]
        C = line1[2, i]  # 极线的参数: Ax + By + C =0; ==> y = (-A/B)x - (C/B)
        W = im1.shape[1]    # 图片width，
        y1 = -C/B       # (0,y1)
        y2 = -(A * W + C) / B   # (W, y2)
        plt.plot([0, W], [y1, y2], 'r')     # 画出每一条极线
        plt.plot([points1[i, 0]], [points1[i, 1]], "b*")    # 画出 每个极点的（x，y）坐标
    plt.imshow(im1, cmap='gray')

    # 第二幅图片
    plt.subplot(1, 2, 2)
    line2 = F.dot(points1.T)
    N2 = line2.shape[1]
    for i in range(N1):
        A = line2[0, i]
        B = line2[1, i]
        C = line2[2, i]  # 极线的参数: Ax + By + C =0; ==> y = (-A/B)x - (C/B)
        W = im1.shape[1]    # 图片width，
        y1 = -C/B       # (0,y1)
        y2 = -(A * W + C) / B   # (W, y2)
        plt.plot([0, W], [y1, y2], 'r')     # 画出每一条极线
        plt.plot([points2[i, 0]], [points2[i, 1]], "b*")    # 画出 每个极点的（x，y）坐标
    plt.imshow(im2, cmap='gray')







'''
COMPUTE_DISTANCE_TO_EPIPOLAR_LINES  computes the average distance of a set a 
points to their corresponding epipolar lines
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    average_distance - the average distance of each point to the epipolar line
'''
def compute_distance_to_epipolar_lines(points1, points2, F):
    # F.Tp2 = l, 求得p2到p1面上的映射直线
    line = F.T.dot(points2.T)  # 3 * N

    dis_sum = 0
    N = points1.shape[0]

    for i in range(N):
        x = points1[i, 0]
        y = points1[i, 1]
        A = line[0, i]
        B = line[1, i]
        C = line[2, i]
        dis_sum += np.abs(A*x + B*y + C) / np.sqrt(A**2 + B**2)

    return dis_sum / N      # 平均距离








if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set2']:
        print ('-'*80)
        print ("Set:", im_set)
        print ('-'*80)

        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
        points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
        assert (points1.shape == points2.shape)

        # Running the linear least squares eight point algorithm
        F_lls = lls_eight_point_alg(points1, points2)
        print ("Fundamental Matrix from LLS  8-point algorithm:\n", F_lls)

        print ("Distance to lines in image 1 for LLS:",
            compute_distance_to_epipolar_lines(points1, points2, F_lls) )
        print ("Distance to lines in image 2 for LLS:",
            compute_distance_to_epipolar_lines(points2, points1, F_lls.T))


        # Running the normalized eight point algorithm
        F_normalized = normalized_eight_point_alg(points1, points2)

        pFp = [points2[i].dot(F_normalized.dot(points1[i]))
            for i in range(points1.shape[0])]
        print ("p'^T F p =", np.abs(pFp).max())
        print ("Fundamental Matrix from normalized 8-point algorithm:\n", \
            F_normalized)
        print ("Distance to lines in image 1 for normalized:", \
            compute_distance_to_epipolar_lines(points1, points2, F_normalized))
        print ("Distance to lines in image 2 for normalized:", \
            compute_distance_to_epipolar_lines(points2, points1, F_normalized.T))
        #
        # Plotting the epipolar lines
        plt.figure(1)
        # plt.title("LLS Eight-Points")
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_lls)
        plt.suptitle("Dataset%s: LLS Eight-Points " %im_set)

        plt.figure(2)
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_normalized)
        plt.suptitle("Dataset%s: Normalized Eight-Points"%im_set)
        plt.show()
