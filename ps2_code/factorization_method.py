import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib.gridspec as gridspec
from epipolar_utils import *

'''
FACTORIZATION_METHOD The Tomasi and Kanade Factorization Method to determine
the 3D structure of the scene and the motion of the cameras.
Arguments:
    points_im1 - N points in the first image that match with points_im2
    points_im2 - N points in the second image that match with points_im1

    Both points_im1 and points_im2 are from the get_data_from_txt_file() method
Returns:
    structure - the structure matrix
    motion - the motion matrix
'''
def factorization_method(points_im1, points_im2):
    N = points_im1.shape[0]
    points_sets = [points_im1, points_im2]      # 2* (N * 3)

    # 建立D矩阵
    D = np.zeros((4, N))
    for i in range(len(points_sets)):   # len : 2
        points = points_sets[i]        # N * 3
        # 中心化点
        centroid = 1.0 / N * points.sum(axis=0)     # 均值 (x,y),
        points[:, 0] -= centroid[0] * np.ones(N)    # x
        points[:, 1] -= centroid[1] * np.ones(N)    # y
        D[2*i:2*i+2, :] = points[:, 0:2].T    # 每一副图片的(x,y)复制到D中

    # svd分解D矩阵
    u, s, vt = np.linalg.svd(D)
    print(u.shape, s.shape, vt.shape)
    print(s)
    M = u[:, 0:3]      # Motion
    S = np.diag(s)[0:3, 0:3].dot(vt[0:3, :])        # structure
    return S, M





if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set1_subset']:
        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points_im1 = get_data_from_txt_file(im_set + '/pt_2D_1.txt')
        points_im2 = get_data_from_txt_file(im_set + '/pt_2D_2.txt')
        points_3d = get_data_from_txt_file(im_set + '/pt_3D.txt')
        assert (points_im1.shape == points_im2.shape)

        # Run the Factorization Method
        structure, motion = factorization_method(points_im1, points_im2)

        # Plot the structure
        fig = plt.figure()
        ax = fig.add_subplot(121, projection = '3d')    # 3d 图
        scatter_3D_axis_equal(structure[0,:], structure[1, :], structure[2, :], ax)  # 散点图
        ax.set_title('Factorization Method')
        ax = fig.add_subplot(122, projection = '3d')
        scatter_3D_axis_equal(points_3d[:,0], points_3d[:,1], points_3d[:,2], ax)
        ax.set_title('Ground Truth')
        plt.suptitle("data-%s"%im_set)
        plt.show()
