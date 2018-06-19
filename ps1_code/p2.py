# CS231A Homework 1, Problem 2
import numpy as np
import matplotlib.pyplot as plt

'''
COMPUTE_CAMERA_MATRIX
Arguments:
     real_XY - Each row corresponds to an actual point on the 2D plane
     front_image - Each row is the pixel location in the front image where Z=0
     back_image - Each row is the pixel location in the back image where Z=150
Returns:
    camera_matrix - The calibrated camera matrix (3x4 matrix)
'''
def compute_camera_matrix(real_XY, front_image, back_image):
    img_num1 = front_image.shape[0]
    img_num2 = back_image.shape[0]

    # 建立真实XYZ的齐次矩阵
    ones = np.ones((img_num1, 1))
    front_z = np.zeros((img_num1, 1))
    front_scence = np.c_[real_XY, front_z, ones]  # 12 * 4

    back_z = 150 * np.ones((img_num2, 1))
    back_scence = np.c_[real_XY, back_z, ones]  # 12 * 4

    # 合并两个真实场景
    M_scene = np.r_[front_scence, back_scence]  # 24 * 4

    # 系数矩阵 A  2n * 8
    n = img_num1 + img_num2
    A = np.zeros((2 * n, 8))
    for i in range(0, A.shape[0], 2):
        idx = int(i / 2)
        A[i, :] = np.hstack((M_scene[idx, :], [0, 0, 0, 0]))
        A[i + 1, :] = np.hstack(([0, 0, 0, 0], M_scene[idx, :]))

    # 图片对应点矩阵 2n * 1
    # b = [U1,V1, U2,V2,..., Un,Vn]
    b = front_image[0].T
    for i in range(1, img_num1, 1):
        b = np.hstack((b, front_image[i].T))
    for j in range(img_num2):
        b = np.hstack((b, back_image[j].T))
    b = np.reshape(b, (2 * n, 1))

    # 计算矩阵，添加最后一行
    # p = np.linalg.lstsq(A, b, rcond=None)  # 直接计算 AM= b ==> M = A^(-1)*b
    # camera_matrix = p[0]

    camera_matrix = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)  # 使用最小二乘计算结果
    camera_matrix = np.reshape(camera_matrix, (2, -1))  # m(8,1) ==> m(2,4)
    camera_matrix = np.vstack((camera_matrix, [0, 0, 0, 1]))  # 添加最后一列 ==> m（3,4）

    return camera_matrix


'''
RMS_ERROR
Arguments:
     camera_matrix - The camera matrix of the calibrated camera
     real_XY - Each row corresponds to an actual point on the 2D plane
     front_image - Each row is the pixel location in the front image where Z=0
     back_image - Each row is the pixel location in the back image where Z=150
Returns:
    rms_error - The root mean square error of reprojecting the points back
                into the images
'''
def rms_error(camera_matrix, real_XY, front_image, back_image):
    img_num1 = front_image.shape[0]
    img_num2 = back_image.shape[0]

    ones = np.ones((img_num1,1))
    # 建立图像XYZ的齐次矩阵
    front_img = np.c_[front_image, ones]     # front_image : n * 3
    back_img = np.c_[back_image, ones]   # back_image: n * 3
    img = np.r_[front_img, back_img]
    img = img.T  # img : 3 * 2n

    # 建立真实XYZ的齐次矩阵
    front_z = np.zeros((img_num1, 1))
    front_scence = np.c_[real_XY, front_z, ones]  # n * 4

    back_z =150 * np.ones((img_num2,1))
    back_scence = np.c_[real_XY, back_z, ones]  # n * 4

    # 合并两个真实场景
    M_scene = np.r_[front_scence, back_scence]  # 2n * 4
    M_scene = M_scene.T  # real : 4 * 2n

    M_scene_trans = camera_matrix.dot(M_scene)  # 变换
    diff_sqr = (M_scene_trans - img)**2 # 平方差
    diff_sum = np.sum(np.sum(diff_sqr,axis=0))  # 平方差的和，先行相加，在列相加
    diff_sum /= (img_num1 + img_num2)   # 求均值

    rms_error = np.sqrt(diff_sum)
    return rms_error


if __name__ == '__main__':
    # Loading the example coordinates setup
    real_XY = np.load('real_XY.npy')
    front_image = np.load('front_image.npy')
    back_image = np.load('back_image.npy')

    print("xy shape:", real_XY.shape)
    print("front_img shape:", front_image.shape)
    print("back_img shape:", back_image.shape)

    camera_matrix = compute_camera_matrix(real_XY, front_image, back_image)
    rmse = rms_error(camera_matrix, real_XY, front_image, back_image)

    print('camera matrix:\n',camera_matrix)
    print()
    print("RMS Error: ", rmse)