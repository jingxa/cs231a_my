import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from imageio import imread
from scipy.spatial.distance import cdist



def kmeans_segmentation(im, features, num_clusters):
    H, W = im.shape[0], im.shape[1]
    N = features.shape[0]

    # 第一步： 随机选择num_clusters个种子
    center_idx = np.random.randint(N, size=num_clusters)
    centriods = features[center_idx]

    matrixes = np.zeros((H, W))
    # 第二步： 迭代器划分
    while True:
        # 每个像素到cneter的距离
        dist = np.zeros((N, num_clusters))

        for i in range(num_clusters):
            dist[:, i] = np.linalg.norm(features - centriods[i, :], axis=1)     # 距离

        # 寻找最近中心
        nearest = np.argmin(dist, axis=1)       # (N,1)
        # 更新
        prev_centriods = centriods

        for i in range(num_clusters):
            pixels_idx = np.where(nearest == i)      # 和 第 i 个中心邻近的像素集合
            cluster = features[pixels_idx]            # (M,5)
            centriods[i, :] = np.mean(cluster, axis=0)      # 重新计算平均值

        # 收敛
        if np.array_equal(prev_centriods, centriods):
            break

    pixels_clusters = np.reshape(nearest, (H, W))
    return pixels_clusters


def meanshift_segmentation(im, features, bandwidth):
    H, W = im.shape[0], im.shape[1]

    N, M = features.shape       # 数量， 特征维度
    mask = np.ones(N)

    clusters = []

    while np.sum(mask) > 0 :    # 当前还有像素未被遍历
        loc = np.argwhere(mask > 0)
        idx = loc[int(np.random.choice(loc.shape[0], 1)[0])][0]     # 随扈挑选一个像素

        mask[idx] = 0   # 标记

        current_mean = features[idx]
        prev_mean = current_mean

        while True:
            dist = np.linalg.norm(features - prev_mean, axis=1)
            incircle = dist < bandwidth # 距离小于半径的点
            mask[incircle] = 0

            current_mean = np.mean(features[incircle], axis=0)  # 新的中心
            # 稳定，收敛
            if np.linalg.norm(current_mean - prev_mean) < 0.01 * bandwidth:
                break
            prev_mean = current_mean

        isValid = True
        for cluster in clusters:
            if np.linalg.norm(cluster - current_mean) < 0.5 * bandwidth:   # 两个划分为一个cluster
                isValid = False

        if isValid:     # 添加一个新cluster
            clusters.append(current_mean)


    pixels_clusters = np.zeros((H, W))

    clusters = np.array(clusters)

    for i in range(N):     # 计算每个像素点的最近中心
        idx = np.argmin(np.linalg.norm(features[i, :] - clusters, axis=1))
        h = int(i/W)
        w = i % W
        pixels_clusters[h, w] = idx

    return  pixels_clusters.astype(int)






def draw_clusters_on_image(im, pixel_clusters):
    num_clusters = int(pixel_clusters.max()) + 1
    average_color = np.zeros((num_clusters, 3))
    cluster_count = np.zeros(num_clusters)

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            c = pixel_clusters[i,j]
            cluster_count[c] += 1
            average_color[c, :] += im[i, j, :]

    for c in range(num_clusters):
        average_color[c,:] /= float(cluster_count[c])
        
    out_im = np.zeros_like(im)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            c = pixel_clusters[i,j]
            out_im[i,j,:] = average_color[c,:]

    return out_im


if __name__ == '__main__':
    
    # Change these parameters to see the effects of K-means and Meanshift
    num_clusters = [5]
    bandwidths = [0.3]


    for filename in ['lake', 'rocks', 'plates']:
        img = imread('data/%s.jpeg' % filename) 

        # Create the feature vector for the images
        features = np.zeros((img.shape[0] * img.shape[1], 5))
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                features[row*img.shape[1] + col, :] = np.array([row, col, 
                    img[row, col, 0], img[row, col, 1], img[row, col, 2]])  #
        features_normalized = features / features.max(axis = 0)

        # Part I: Segmentation using K-Means
        # for nc in num_clusters:
        #     clustered_pixels = kmeans_segmentation(img, features_normalized, nc)
        #     cluster_im = draw_clusters_on_image(img, clustered_pixels)
        #     plt.imshow(cluster_im)
        #     plt.title('K-means with %d clusters on %s.jpeg' % (int(nc), filename))
        #     plt.show()


        # # Part II: Segmentation using Meanshift
        for bandwidth in bandwidths:
            clustered_pixels = meanshift_segmentation(img, features_normalized, bandwidth)
            cluster_im = draw_clusters_on_image(img, clustered_pixels)
            plt.imshow(cluster_im)
            plt.title('Meanshift with bandwidth %.2f on %s.jpeg' % (bandwidth, filename))
            plt.show()
