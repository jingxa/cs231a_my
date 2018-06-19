# CS231A Homework 1, Problem 3


import numpy as np
from utils import mat2euler
import math
import matplotlib.pyplot as plt

'''
COMPUTE_VANISHING_POINTS
Arguments:
    points - 四对点，前两对为一条直线，后两对为一条，两条线是平行的；
Returns:
    灭点
'''
def compute_vanishing_point(points):
    # 获取四对点
    x1 = points[0, 0]
    y1 = points[0, 1]   # (x1,y1)
    x2 = points[1, 0]
    y2 = points[1, 1]   # (x2,y2)
    x3 = points[2, 0]
    y3 = points[2, 1]   # （x3,y3）
    x4 = points[3, 0]
    y4 = points[3, 1]   # (x4,y4)

    # 计算两条直线的参数
    a1 = (y2 - y1)/(x2 - x1)
    b1 = y1 - a1 * x1

    a2 = (y4 - y3)/(x4 - x3)
    b2 = y3 - a2 * x3

    # 计算交点
    x = (b1 - b2)/(a2 - a1)
    y = a1 * x + b1
    vanish_point = np.array([x, y])
    return vanish_point
    pass



'''
COMPUTE_K_FROM_VANISHING_POINTS
Arguments:
    vanishing_points - a list of vanishing points

Returns:
    K - the intrinsic camera matrix (3x3 matrix)
'''
def compute_K_from_vanishing_points(vanishing_points):
    v1 = vanishing_points[0]
    v2 = vanishing_points[1]
    v3 = vanishing_points[2]

    # 构建系数矩阵
    A = np.zeros((3, 4))
    A[0] = np.array([(v1[0]*v2[0] + v1[1]*v2[1]), (v1[0] + v2[0]), (v1[1] + v2[1]), 1])
    A[1] = np.array([(v1[0]*v3[0] + v1[1]*v3[1]), (v1[0] + v3[0]), (v1[1] + v3[1]), 1])
    A[2] = np.array([(v2[0]*v3[0] + v2[1]*v3[1]), (v2[0] + v3[0]), (v2[1] + v3[1]), 1])

    # print("A:\n",A)
    # SVD分解
    U, s, vT = np.linalg.svd(A, full_matrices=True)

    # print('SVD:\n',U,U.shape)
    # print('s:\n',s, s.shape)
    # print('v:\n',vT,vT.shape)
    # print()

    w = vT[-1, :]   # 取最后一行，最为最优解
    print('w:\n',w, w.shape)
    omega = np.array([  # 建立w矩阵
        [w[0], 0, w[1]],
        [0, w[0], w[2]],
        [w[1], w[2], w[3]]
    ], dtype=np.float64)

    print()
    # 使用cholesky 分解得到K
    kT_inv = np.linalg.cholesky(omega)  # w = (k*k.T)^-1 ==> 分解为 k.T^-1
    k = np.linalg.inv(kT_inv.T)
    k /= k[2, 2]    # 最小化
    return k



'''
COMPUTE_K_FROM_VANISHING_POINTS
Arguments:
    vanishing_pair1 - a list of a pair of vanishing points computed from lines within the same plane
    vanishing_pair2 - a list of another pair of vanishing points from a different plane than vanishing_pair1
    K - the camera matrix used to take both images

Returns:
    angle - the angle in degrees between the planes which the vanishing point pair comes from2
'''
def compute_angle_between_planes(vanishing_pair1, vanishing_pair2, K):
    omega_inv = K.dot(K.T)

    # a set of vanishing points on one plane
    v1 = np.hstack((vanishing_pair1[0], 1))
    v2 = np.hstack((vanishing_pair1[1], 1))

    # another set of vanishing points on the other plane
    v3 = np.hstack((vanishing_pair2[0], 1))
    v4 = np.hstack((vanishing_pair2[1], 1))

    # find two vanishing lines
    L1 = np.cross(v1.T, v2.T)  # 为什么如此？
    L2 = np.cross(v3.T, v4.T)

    # find the angle between planes
    costheta = (L1.T.dot(omega_inv).dot(L2)) / (np.sqrt(L1.T.dot(omega_inv).dot(L1)) * np.sqrt(L2.T.dot(omega_inv).dot(L2)))
    theta = (np.arccos(costheta) / math.pi) * 180

    return theta

'''
COMPUTE_K_FROM_VANISHING_POINTS
Arguments:
    vanishing_points1 - a list of vanishing points in image 1
    vanishing_points2 - a list of vanishing points in image 2
    K - the camera matrix used to take both images

Returns:
    R - the rotation matrix between camera 1 and camera 2
'''
def compute_rotation_matrix_between_cameras(vanishing_points1, vanishing_points2, K):

    ones = np.ones((vanishing_points1.shape[0], 1))
    # 建立齐次矩阵
    v1 = np.hstack((vanishing_points1, ones)).T
    v2 = np.hstack((vanishing_points2, ones)).T

    # 计算真实平行线的方向
    # d = K^-1 * v
    k_inv = np.linalg.inv(K)
    D1 = k_inv.dot(v1) / np.linalg.norm(k_inv.dot(v1),axis=0)   # 按列计算norm范数
    D2 = k_inv.dot(v2) / np.linalg.norm(k_inv.dot(v2),axis=0)

    # d2 = R * d1, d2.T = d1.T * R.T
    R = np.linalg.lstsq(D1.T, D2.T,rcond = None)[0].T

    return R



if __name__ == '__main__':

    # Part A: Compute vanishing points
    v1 = compute_vanishing_point(np.array([[674,1826],[2456,1060],[1094,1340],[1774,1086]]))
    v2 = compute_vanishing_point(np.array([[674,1826],[126,1056],[2456,1060],[1940,866]]))
    v3 = compute_vanishing_point(np.array([[1094,1340],[1080,598],[1774,1086],[1840,478]]))

    v1b = compute_vanishing_point(np.array([[314,1912],[2060,1040],[750,1378],[1438,1094]]))
    v2b = compute_vanishing_point(np.array([[314,1912],[36,1578],[2060,1040],[1598,882]]))
    v3b = compute_vanishing_point(np.array([[750,1378],[714,614],[1438,1094],[1474,494]]))

    print('----------------')
    # Part B: Compute the camera matrix
    vanishing_points = [v1, v2, v3]
    print("Intrinsic Matrix:\n",compute_K_from_vanishing_points(vanishing_points))

    K_actual = np.array([[2448.0, 0, 1253.0],[0, 2438.0, 986.0],[0,0,1.0]])
    print()
    print("Actual Matrix:\n", K_actual)
    print('-------------------------------')

    # Part D: Estimate the angle between the box and floor
    floor_vanishing1 = v1
    floor_vanishing2 = v2
    box_vanishing1 = v3
    box_vanishing2 = compute_vanishing_point(np.array([[1094,1340],[1774,1086],[1080,598],[1840,478]]))
    angle = compute_angle_between_planes([floor_vanishing1, floor_vanishing2], [box_vanishing1, box_vanishing2], K_actual)
    print()
    print("Angle between floor and box:", angle)


    # Part E: Compute the rotation matrix between the two cameras
    rotation_matrix = compute_rotation_matrix_between_cameras(np.array([v1, v2, v3]), np.array([v1b, v2b, v3b]), K_actual)
    print()
    print("Rotation between two cameras:\n", rotation_matrix)
    z,y,x = mat2euler(rotation_matrix)
    print()
    print("Angle around z-axis (pointing out of camera): %f degrees" % (z * 180 / math.pi))
    print("Angle around y-axis (pointing vertically): %f degrees" % (y * 180 / math.pi))
    print("Angle around x-axis (pointing horizontally): %f degrees" % (x * 180 / math.pi))
