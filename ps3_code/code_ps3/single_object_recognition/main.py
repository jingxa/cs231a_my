import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random
from ps3_code.code_ps3.single_object_recognition.utils import *
import math

from collections import defaultdict



def match_keypoints(descriptors1, descriptors2, threshold = 0.7):
    matches_list = []

    N = descriptors1.shape[0]
    for i in range(N):
        des1 = descriptors1[i]      # 一个点的描述符
        dist = np.sqrt(np.sum((descriptors2 - des1)**2, axis=1))        # 当前点和另一幅图片所有点的欧式距离
        index_sort = np.argsort(dist)           # 按照大小排序，返回序号

        closed = index_sort[0]
        second_closed = index_sort[1]

        if(dist[closed] < threshold * dist[second_closed]):
            matches_list.append([i, closed])

    matches = np.array(matches_list)
    return matches



def refine_match(keypoints1, keypoints2, matches, reprojection_threshold = 10,
        num_iterations = 1000):

    best_model = None       # 最佳模型
    best_inliers = []       # 内点集合
    best_count = 0          # 最佳匹配对数

    sample_size = 4         # H 是 9*9矩阵，除去尺度，有八个自由度，需要4对对应点
    P = np.zeros((2 * sample_size, 9))      # 建立 2N*9的矩阵，其中每个对应对建立两个等式
    N = matches.shape[0]

    for i in range(num_iterations):
        sample_indexes = random.sample(range(0, N), sample_size)     # 随机抽取4对样本
        sample = matches[sample_indexes, :]

        for index, elem in enumerate(sample):       # 获得序列和数据
            # 取一对对应点
            p1_idx = elem[0]
            p2_idx = elem[1]

            # 转化为齐次坐标系
            point1 = keypoints1[p1_idx, 0:2]        # u,v 坐标
            point1 = np.append(point1, 1)           # (u, v, 1)

            point2 = keypoints2[p2_idx, 0:2]
            u = point2[0]
            v = point2[1]

            # 建立 P 矩阵
            P[2 * index, :] = np.reshape(np.array([point1, np.zeros(3), -u * point1]), -1)
            P[2 * index + 1, :] = np.reshape(np.array([np.zeros(3), point1, -v * point1]), -1)

        # 求解当前的H矩阵
        U, s, VT = np.linalg.svd(P)
        H = VT[-1, :].reshape(3, 3)
        H /= H[2, 2]        # 归一化

        inliers = []        # 当前对应的 内点
        count = 0

        for index, match in enumerate(matches):     # 对每一对对应点进行变换评估
            p1 = keypoints1[match[0], 0:2]      # u,v
            p1 = np.append(p1, 1)               # (u,v,1)

            p2_pred = H.dot(p1)             # p1 变换后的点
            if p2_pred[2] != 0:
                p2_pred /= p2_pred[2]           # 归一化
            p2_pred = p2_pred[0:2]          # u,v

            p2 = keypoints2[match[1], 0:2]
            err = np.sqrt( np.sum( (p2 - p2_pred) ** 2 ) )
            if err < reprojection_threshold:        # 此对应点为正确
                count += 1
                inliers.append(index)

        # 记录最佳 H
        if count > best_count:
            best_model = H
            best_inliers = inliers
            best_count = count

    return best_inliers, best_model


def get_object_region(keypoints1, keypoints2, matches, obj_bbox, thresh = 4,
        nbins = 4):

    cx, cy, w, h, orient = [], [], [], [], []

    # 第一步： 计算边界盒子数组
    for match in matches:
        p1_idx = match[0]
        p2_idx = match[1]

        p1 = keypoints1[p1_idx]
        p2 = keypoints2[p2_idx]

        u1, v1, s1, theta1 = p1[0], p1[1], p1[2], p1[3]     # 两个点的成员
        u2, v2, s2, theta2 = p2[0], p2[1], p2[2], p2[3]

        # 寻找对应点在img2 中的边界盒子，采用旋转平移进行计算
        xmin, ymin, xmax, ymax = obj_bbox
        xc1 = (xmin + xmax) / 2.0
        yc1 = (ymin + ymax) / 2.0       # 中心点的坐标
        w1 = (xmax - xmin) * 1.0
        h1 = (ymax - ymin) * 1.0        # 使用浮点表示

        O2 = theta2 - theta1
        xc2 = (s2/s1) * np.cos(O2) * (xc1 - u1) - (s2/s1) * np.sin(O2) * (yc1 - v1) + u2
        yc2 = (s2/s1) * np.sin(O2) * (xc1 - u1) + (s2/s1) * np.cos(O2) * (yc1 - v1) + v2
        w2 = (s2/s1) * w1
        h2 = (s2/s1) * h1       # 缩放

        # 保存到数组中
        cx.append(xc2)
        cy.append(yc2)
        w.append(w2)
        h.append(h2)
        orient.append(O2)       # 这个方向有点不理解

    # 第二步： 计算盒子的子网格划分
    cx_min, cx_max = min(cx), max(cx)
    cy_min, cy_max = min(cy), max(cy)
    w_min, w_max = min(w), max(w)
    h_min, h_max = min(h), max(h)
    orient_min, orient_max = min(orient), max(orient)

    cx_bin_size = (cx_max - cx_min) / float(nbins)
    cy_bin_size = (cy_max - cy_min) / float(nbins)
    w_bin_size = (w_max - w_min) / float(nbins)
    h_bin_size = (h_max - h_min) / float(nbins)
    orient_bin_size = (orient_max - orient_min) / float(nbins)


    # 第三步： 统计每个子网格的计数
    bins = defaultdict(list)        # 由于nbins为4，那么就只计算4个参数
    N = matches.shape[0]
    for n in range(N):
        x_center = cx[n]
        y_center = cy[n]
        w_center = w[n]
        orient_center = orient[n]

        for i in range(nbins):
            for j in range(nbins):
                    for k in range(nbins):
                            for l in range(nbins):
                                if(cx_min + i * cx_bin_size <= x_center
                                        and x_center <= cx_min +(i+1) * cx_bin_size):       # x坐标
                                            if(cy_min + j * cy_bin_size <= y_center
                                                and y_center <= cy_min + (j+1) * cy_bin_size):
                                                    if(w_min + k * w_bin_size <= w_center
                                                        and w_center <= w_min + (k+1)*w_bin_size):
                                                            if(orient_min + l*orient_bin_size <= orient_center
                                                                and orient_center <= orient_min + (l+1) * orient_bin_size):
                                                                    bins[(i, j, k, l)].append(n)
    # 第四步： 统计
    cx0, cy0, w0, h0, orient0 = [], [], [], [], []

    for bin_idx in bins:
        indices = bins[bin_idx]
        votes = len(indices)

        if(votes >= thresh):
            cx0.append(np.sum(np.array(cx)[indices]) / votes)       # 平均数
            cy0.append(np.sum(np.array(cy)[indices]) / votes)
            w0.append(np.sum(np.array(w)[indices]) / votes)
            h0.append(np.sum(np.array(w)[indices]) / votes)
            orient0.append(np.sum(np.array(orient)[indices]) / votes)

    return cx0, cy0, w0, h0, orient0


















'''
MATCH_OBJECT: The pipeline for matching an object in one image with another

Arguments:
    im1 - The first image read in as a ndarray of size (H, W, C).

    descriptors1 - Descriptors corresponding to the first image. Each row
        corresponds to a descriptor. This is a ndarray of size (M_1, 128).

    keypoints1 - Keypoints in the first image. Each row is a SIFT keypoint
        consisting of (u, v, scale, theta). Overall, this variable is a ndarray
        of size (M_1, 4).

    im2 - The second image read in as a ndarray of size (H, W, C).

    descriptors2 - Descriptors corresponding to the second image. Each row
        corresponds to a descriptor. This is a ndarray of size (M_2, 128).

    keypoints2 - Keypoints in the second image. Each row is a SIFT keypoint
        consisting of (u, v, scale, theta). Overall, this variable is a ndarray
        of size (M_2, 4).

    obj_bbox - An ndarray of size (4,) that contains [xmin, ymin, xmax, ymax]
        of the bounding box. Note that the point (xmin, ymin) is one corner of
        the box and (xmax, ymax) is the opposite corner of the box.

Returns:
    descriptors - The descriptors corresponding to the keypoints inside the
        bounding box.

    keypoints - The pixel locations of the keypoints that reside in the 
        bounding box
'''
def match_object(im1, descriptors1, keypoints1, im2, descriptors2, keypoints2,
        obj_bbox):
    # Part A
    descriptors1, keypoints1, = select_keypoints_in_bbox(descriptors1,
        keypoints1, obj_bbox)
    matches = match_keypoints(descriptors1, descriptors2)
    plot_matches(im1, im2, keypoints1, keypoints2, matches)
    
    # Part B
    inliers, model = refine_match(keypoints1, keypoints2, matches)
    plot_matches(im1, im2, keypoints1, keypoints2, matches[inliers,:])

    # Part C
    cx, cy, w, h, orient = get_object_region(keypoints1, keypoints2,
        matches[inliers,:], obj_bbox)

    plot_bbox(cx, cy, w, h, orient, im2)


if __name__ == '__main__':
    # Load the data
    data = sio.loadmat('SIFT_data.mat')
    images = data['stopim'][0]
    obj_bbox = data['obj_bbox'][0]
    keypoints = data['keypt'][0]
    descriptors = data['sift_desc'][0]
    
    np.random.seed(0)

    # for i in [2, 1, 3, 4]:

    for i in [4]:
        match_object(images[0], descriptors[0], keypoints[0], images[i],
            descriptors[i], keypoints[i], obj_bbox)
