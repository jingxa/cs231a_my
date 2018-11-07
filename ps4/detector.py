import numpy as np
import os
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from imageio import imread
from utils import *


def run_detector(im, clf, window_size, cell_size, block_size, nbins, thresh=1):
    H, W = im.shape
    win_H, win_W = window_size
    stride = int(block_size * cell_size / 2)

    # 返回值
    bboxes = []
    scores = []

    # 计算每个窗口是否为face
    for i in range(0, W - win_W, stride):
        for j in range(0, H - win_H, stride):
            bbox = [i, j, win_W, win_H]     # 窗口 [xmin ymin width height]
            im_bbox = im[j:j+win_H, i: i+win_W]
            feature_im = compute_hog_features(im_bbox, cell_size, block_size, nbins)        # HoG
            score_im = clf.decision_function(feature_im.flatten().reshape(1,-1))        # 先将HoG特征变为vector,然后进行判断是否为face特征

            if score_im > thresh:
                scores.append(score_im)
                bboxes.append(bbox)

    # 变换为numpy类型
    bboxes = np.array(bboxes)
    scores = np.array(scores)

    return bboxes, scores


def non_max_suppression(bboxes, confidences):
    nms_bboxs = []   # 返回窗口

    # 对confidences 排序
    con_idx = np.argsort(-confidences.flatten())          # 从大到小排列

    N = bboxes.shape[0]

    for i in range(N):
        bbox = bboxes[con_idx[i], :]        # 一个窗口,[xmin, ymin, width, height]

        if i == 0:      # 第一个窗口，加入结果序列
            nms_bboxs.append(bbox)
        else:           # 查看当前窗口是否和已有窗口重叠

            # 计算窗口中心
            cx = (2 * bbox[0] + bbox[2]) / 2
            cy = (2 * bbox[1] + bbox[3]) / 2

            isOverlap = False

            for j in range(len(nms_bboxs)):
                xmin, ymin, w, h = nms_bboxs[j][0], nms_bboxs[j][1],\
                                   nms_bboxs[j][2], nms_bboxs[j][3]

                xmax, ymax = (xmin + w), (ymin + h)

                if xmin <= cx <= xmax and ymin <= cy <= ymax:     # 两个窗口重叠
                    isOverlap = True
                    break

            if not isOverlap:       # 没有重叠
                nms_bboxs.append(bbox)

    nms_bboxs = np.array(nms_bboxs)
    return nms_bboxs





if __name__ == '__main__':
    block_size = 2
    cell_size = 6
    nbins = 9
    window_size = np.array([36, 36])

    # compute or load features for training
    if not (os.path.exists('data/features_pos.npy') and os.path.exists('data/features_neg.npy')):
        features_pos = get_positive_features('data/caltech_faces/Caltech_CropFaces', cell_size, window_size, block_size, nbins)
        num_negative_examples = 10000
        features_neg = get_random_negative_features('data/train_non_face_scenes', cell_size, window_size, block_size, nbins, num_negative_examples)
        np.save('data/features_pos.npy', features_pos)
        np.save('data/features_neg.npy', features_neg)
    else:
        features_pos = np.load('data/features_pos.npy')     # face特征
        features_neg = np.load('data/features_neg.npy')     # non-face特征

    X = np.vstack((features_pos, features_neg))
    Y = np.hstack((np.ones(len(features_pos)), np.zeros(len(features_neg))))

    # Train the SVM
    clf = LinearSVC(C=1, tol=1e-6, max_iter=10000, fit_intercept=True, loss='hinge')
    clf.fit(X, Y)
    score = clf.score(X, Y)

    # Part A: Sliding window detector
    im = imread('data/people.jpg').astype(np.uint8)
    bboxes, scores = run_detector(im, clf, window_size, cell_size, block_size, nbins)
    plot_img_with_bbox(im, bboxes, 'Without nonmaximal suppresion')
    plt.show()

    # Part B: Nonmaximal suppression
    bboxes = non_max_suppression(bboxes, scores)
    plot_img_with_bbox(im, bboxes, 'With nonmaximal suppresion')
    plt.show()
