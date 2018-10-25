import numpy as np
import skimage.io as sio
from scipy.io import loadmat
from ps3_code.code_ps3.hog.plotting import *
import math


def compute_gradient(im):
    H, W = im.shape

    angles = np.zeros((H-2, W-2))
    magnitudes = np.zeros((H-2, W-2))       # 建立两个返回值矩阵

    for i in range(1, H-1):
        for j in range(1, W-1):     # 最外圈的点不计算
            top = im[i-1, j]
            down = im[i+1, j]
            left = im[i, j-1]
            right = im[i, j+1]      # 上下左右

            angle = np.arctan2(top - down, left - right) * (180 / math.pi)      # 转化为度
            magn = np.sqrt((top-down)**2 + (left - right)**2)

            if(angle < 0):      # 加180
                angle += 180

            angles[i-1, j-1] = angle
            magnitudes[i-1, j-1] = magn

    return angles, magnitudes


def generate_histogram(angles, magnitudes, nbins = 9):
    histogram = np.zeros(nbins)

    bin_size = 180 / nbins
    center_angles = np.zeros_like(histogram)

    for i in range(nbins):
        center_angles[i] = (0.5 + i) * bin_size     # 计算每个区间中心

    M, N = angles.shape

    for m in range(M):
        for n in range(N):
            angle = angles[m, n]
            magn = magnitudes[m, n]

            abs_diff = np.abs(center_angles - angle)

            # 当 angle 趋于0度
            if (180 - center_angles[-1] + angle) < abs_diff[-1]:        # 更新下最后区间的大小
                abs_diff[-1] = 180 - center_angles[-1] + angle

            # angle趋于180度
            if(180 + center_angles[0] - angle) < abs_diff[0]:
                abs_diff[0] = 180 - angle + center_angles[0]

            # 统计直方图
            bin1, bin2 = np.argsort(abs_diff)[0:2]      # 取最近的两个
            histogram[bin1] += magn * abs_diff[bin2] / (180.0 / nbins)
            histogram[bin2] += magn * abs_diff[bin1] / (180.0 / nbins)

    return histogram



def compute_hog_features(im, pixels_in_cell, cells_in_block, nbins):

    # 第一步： 获得角度和梯度
    angles, magnitudes = compute_gradient(im)

    # 第二步： 划分block 和cell
    cell_size = pixels_in_cell
    block_size = pixels_in_cell * cells_in_block

    # 第三步：计算滑动窗口
    H, W = angles.shape
    stride = int(block_size / 2)
    H_blocks = int((H - block_size) / stride) + 1
    W_blocks = int((W - block_size) / stride) + 1

    # 第四步： 计算每个cell的 histogram
    hog_fe = np.zeros((H_blocks, W_blocks, cells_in_block * cells_in_block * nbins))

    for h in range(H_blocks):
        for w in range(W_blocks):
            block_angles = angles[h*stride: h * stride + block_size,
                           w * stride: w*stride + block_size]      # 一个block的角度
            block_magn = magnitudes[h*stride: h*stride+block_size,
                            w*stride: w*stride+block_size]         # 梯度

            # 将一个block中的每个cell表示为一个方向直方图
            block_hog_fe = np.zeros((cells_in_block, cells_in_block, nbins))

            for i in range(cells_in_block):
                for j in range(cells_in_block):
                        cell_angles = block_angles[i*pixels_in_cell : (i+1)*pixels_in_cell,
                                        j*pixels_in_cell : (j+1)*pixels_in_cell]
                        cell_magns = block_magn[i*pixels_in_cell : (i+1) * pixels_in_cell,
                                        j*pixels_in_cell : (j+1) * pixels_in_cell]

                        cell_hist = generate_histogram(cell_angles, cell_magns, nbins)

                        block_hog_fe[i, j, :] = cell_hist

            # 归一化
            block_hog_fe = np.reshape(block_hog_fe, -1)

            block_hog_fe /= np.linalg.norm(block_hog_fe)

            hog_fe[h, w, :] = block_hog_fe


    return hog_fe

if __name__ == '__main__':
    # # Part A: Checking the image gradient
    # print ('-' * 80)
    # print ('Part A: Image gradient')
    # print ('-' * 80)
    # im = sio.imread('simple.jpg', True)
    # grad_angle, grad_magnitude = compute_gradient(im)
    # print ("Expected angle: 126.339396329")
    # print ("Expected magnitude: 0.423547566786")
    # print ("Checking gradient test case 1:", \
    #     np.abs(grad_angle[0][0] - 126.339396329) < 1e-3 and \
    #     np.abs(grad_magnitude[0][0] - 0.423547566786) < 1e-3)
    #
    # im = np.array([[1, 2, 2, 4, 8],
    #                 [3, 0, 1, 5, 10],
    #                 [10, 13, 12, 2, 7],
    #                 [10, 5, 1, 0, 3],
    #                 [1, 1, 1.5, 2, 2.5]])
    # grad_angle, grad_magnitude = compute_gradient(im)
    # correct_angle = np.array([[ 100.30484647,   63.43494882,  167.47119229],
    #                           [  68.19859051,    0.        ,   45.        ],
    #                           [  53.13010235,   64.53665494,  180.        ]])
    # correct_magnitude = np.array([[ 11.18033989,  11.18033989,   9.21954446],
    #                               [  5.38516481,  11.        ,   7.07106781],
    #                               [ 15.        ,  11.62970335,   2.        ]])
    # print ("Expected angles: \n", correct_angle)
    # print ("Expected magnitudes: \n", correct_magnitude)
    # print ("Checking gradient test case 2:", \
    #     np.allclose(grad_angle, correct_angle) and \
    #     np.allclose(grad_magnitude, correct_magnitude))
    #
    # # Part B: Checking the histogram generation
    # print ('-' * 80)
    # print ('Part B: Histogram generation')
    # print ('-' * 80)
    # angles = np.array([[10, 30, 50], [70, 90, 110], [130, 150, 170]])
    # magnitudes = np.arange(1,10).reshape((3,3))
    # print ("Checking histogram test case 1:", \
    #     np.all(generate_histogram(angles, magnitudes, nbins = 9) == np.arange(1,10)))
    #
    # angles = np.array([[20, 40, 60], [80, 100, 120], [140, 160, 180]])
    # magnitudes = np.arange(1,19,2).reshape((3,3))
    # histogram = generate_histogram(angles, magnitudes, nbins = 9)
    # print ("Checking histogram test case 2:", \
    #     np.all(histogram  == np.array([9, 2, 4, 6, 8, 10, 12, 14, 16])))
    #
    # angles = np.array([[13, 23, 14.3], [53, 108, 1], [77, 8, 32]])
    # magnitudes = np.ones((3,3))
    # histogram = generate_histogram(angles, magnitudes, nbins = 9)
    # print ("Submit these results:", histogram)
    #
    # Part C: Computing and displaying the final HoG features
    # vary cell size to change the output feature vector. These parameters are common parameters
    pixels_in_cell = 8
    cells_in_block = 2
    nbins = 9
    im = sio.imread('car.jpg', True)
    car_hog_feat = compute_hog_features(im, pixels_in_cell, cells_in_block, nbins)
    show_hog(im, car_hog_feat, figsize = (18,6))
