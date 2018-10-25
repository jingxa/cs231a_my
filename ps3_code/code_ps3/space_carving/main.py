import numpy as np
import scipy.io as sio
import argparse
from ps3_code.code_ps3.space_carving.camera import Camera
from ps3_code.code_ps3.space_carving.plotting import *


# A very simple, but useful method to take the difference between the
# first and second element (usually for 2D vectors)
def diff(x):
    return x[1] - x[0]


def form_initial_voxels(xlim, ylim, zlim, num_voxels):
    x_dim = xlim[1] - xlim[0]
    y_dim = ylim[1] - ylim[0]
    z_dim = zlim[1] - zlim[0]   # 获取三轴的长度

    total_vol = x_dim * y_dim * z_dim  # 体积
    one_vol = float(total_vol / num_voxels)
    voxel_size = np.cbrt(one_vol)                   # 三次方根

    x_voxel_num = np.round(x_dim / voxel_size)      # x轴的cube 数量,round 取整
    y_voxel_num = np.round(y_dim / voxel_size)
    z_voxel_num = np.round(z_dim / voxel_size)

    x = np.linspace(xlim[0] + 0.5 * voxel_size,
                    xlim[0] + (0.5 + x_voxel_num - 1) * voxel_size, x_voxel_num )     # 只取前 num_voxels个
    y = np.linspace(ylim[0] + 0.5 * voxel_size,
                    ylim[0] + (0.5 + y_voxel_num - 1) * voxel_size, y_voxel_num)     # 只取前 num_voxels个
    z = np.linspace(zlim[0] + 0.5 * voxel_size,
                    zlim[0] + (0.5 + z_voxel_num - 1) * voxel_size, z_voxel_num)     # 只取前 num_voxels个

    XX, YY, ZZ = np.meshgrid(x, y, z)
    voxels = np.r_[(XX.reshape(-1), YY.reshape(-1), ZZ.reshape(-1))].reshape(3, -1).T   # N *3

    return voxels, voxel_size



def get_voxel_bounds(cameras, estimate_better_bounds = False, num_voxels = 4000):
    camera_positions = np.vstack([c.T for c in cameras])

    print(camera_positions.shape)
    print("0:", camera_positions[0])

    xlim = [camera_positions[:,0].min(), camera_positions[:,0].max()]
    ylim = [camera_positions[:,1].min(), camera_positions[:,1].max()]
    zlim = [camera_positions[:,2].min(), camera_positions[:,2].max()]

    # For the zlim we need to see where each camera is looking. 
    camera_range = 0.6 * np.sqrt(diff( xlim )**2 + diff( ylim )**2)
    for c in cameras:
        viewpoint = c.T - camera_range * c.get_camera_direction()
        zlim[0] = min( zlim[0], viewpoint[2] )
        zlim[1] = max( zlim[1], viewpoint[2] )

    # Move the limits in a bit since the object must be inside the circle
    xlim = xlim + diff(xlim) / 4 * np.array([1, -1])
    ylim = ylim + diff(ylim) / 4 * np.array([1, -1])

    if estimate_better_bounds:
        voxels, voxel_size  = form_initial_voxels(xlim, ylim, zlim, num_voxels)

        for c in cameras:
            voxels = carve(voxels, c)

        min_point = np.min(voxels, axis=0) - voxel_size
        max_point = np.max(voxels, axis=0) + voxel_size

        xlim[0], ylim[0], zlim[0] = min_point[0], min_point[1], min_point[2]
        xlim[1], ylim[1], zlim[1] = max_point[0], max_point[1], max_point[2]
    return xlim, ylim, zlim
    


def carve(voxels, camera):
    N = voxels.shape[0]
    homo_voxels = np.c_[voxels, np.ones((N, 1))].T  # 4 * N
    voxel_index = np.arange(0, N)

    P = camera.P        # 投影矩阵 3 * 4
    img_voxels = P.dot(homo_voxels)   # 3 * N  , 投影到图片
    img_voxels /= img_voxels[2, :]   # 归一化
    img_voxels = img_voxels[0:2, :].T     # 去掉z轴 N*2

    sli = camera.silhouette  # 从camera文件中了解相关信息
    sli_idx = np.nonzero(sli)
    xmin, xmax = np.min(sli_idx[1]), np.max(sli_idx[1])     # 列
    ymin, ymax = np.min(sli_idx[0]), np.max(sli_idx[0])     # 行

    voxelX = img_voxels[:, 0]
    voxelY = img_voxels[:, 1]

    x_filter = np.all([voxelX > xmin, voxelX < xmax], axis=0)       # 一个轴上的逻辑与运算
    y_filter = np.all([voxelY > ymin, voxelY < ymax], axis=0)

    filter = np.all([x_filter, y_filter], axis=0)
    img_voxels = img_voxels[filter, :]      # 过滤大于轮廓矩阵的像素点
    voxel_index = voxel_index[filter]     # 过滤掉序号

    img_voxels = img_voxels.astype(int)     # 由于归一化，可能有小数，转为整数
    sli_filter = (sli[img_voxels[:, 1], img_voxels[:, 0]] == 1)     # (x,y)是否在轮廓矩阵中
    voxel_index = voxel_index[sli_filter]

    return voxels[voxel_index, :]





def estimate_silhouette(im):
    return np.logical_and(im[:,:,0] > im[:,:,2], im[:,:,0] > im[:,:,1] )


if __name__ == '__main__':
    estimate_better_bounds = True
    use_true_silhouette = False
    frames = sio.loadmat('frames.mat')['frames'][0]
    cameras = [Camera(x) for x in frames]

    # Generate the silhouettes based on a color heuristic
    if not use_true_silhouette:
        for i, c in enumerate(cameras):
            c.true_silhouette = c.silhouette
            c.silhouette = estimate_silhouette(c.image)
            if i == 0:
                plt.figure()
                plt.subplot(121)
                plt.imshow(c.true_silhouette, cmap = 'gray')
                plt.title('True Silhouette')
                plt.subplot(122)
                plt.imshow(c.silhouette, cmap = 'gray')
                plt.title('Estimated Silhouette')
                plt.show()

    # Generate the voxel grid
    # You can reduce the number of voxels for faster debugging, but
    # make sure you use the full amount for your final solution
    num_voxels = 6e6
    # # xlim, ylim, zlim = get_voxel_bounds(cameras, estimate_better_bounds)
    # print("lim:", xlim, ylim, zlim)
    # print("--------------------")

    # This part is simply to test forming the initial voxel grid
    # voxels, voxel_size = form_initial_voxels(xlim, ylim, zlim, 4000)
    # plot_surface(voxels)
    # voxels, voxel_size = form_initial_voxels(xlim, ylim, zlim, num_voxels)

    # Test the initial carving
    # voxels = carve(voxels, cameras[0])
    # if use_true_silhouette:
    #     plot_surface(voxels)

    # Result after all carvings
    # for c in cameras:
    #     voxels = carve(voxels, c)
    # plot_surface(voxels, voxel_size)
