# 结果

#### [1.space carving](#1-space-carving)
#### [2.SIFT](#2-sift)
#### [3.HOG](#3-hog)

---

## 1. space carving 

### 1.1 form_initial_voxels形成一个初始立方体

![a](/images/ps3/space_c_a.png)

### 1.2 carving : 一个视角的裁剪

![b](/images/ps3/space_c_b.png)

### 1.3 没有优化边界的多视角裁剪

![c](/images/ps3/space_c_c.png)

### 1.4 优化边界的多视角裁剪

![d](/images/ps3/space_c_d.png)


## 2. SIFT

本节中，主要是进行了(1)关键点匹配，(2)RANSAC优化匹配，(3)Hough Voting 变换 边界盒子，使用了4组测试数据

在边界盒子的参数设置中：

- thresh : 5
- nbins: 4

的条件下， 结果如下：

### 2.1 测试样本1

![sift_a_1](/images/ps3/sift_a_1.png)


![sift_a_2](/images/ps3/sift_a_2.png)


![sift_a_3](/images/ps3/sift_a_3.png)


### 2.2 测试样本2

![sift_b_1](/images/ps3/sift_b_1.png)


![sift_b_2](/images/ps3/sift_b_2.png)


![sift_b_3](/images/ps3/sift_b_3.png)


### 2.3 测试样本3

![sift_c_1](/images/ps3/sift_c_1.png)


![sift_c_2](/images/ps3/sift_c_2.png)


![sift_c_3](/images/ps3/sift_c_3.png)


### 2.4 测试样本4



![sift_d_1](/images/ps3/sift_d_1.png)


![sift_d_2](/images/ps3/sift_d_2.png)


![sift_d_3](/images/ps3/sift_d_3.png)



#### 2.5 测试参数

在修改上述两个参数为：

- thresh: 4
- nbins: 4
的情况下，只有第一和第四测试的边界盒子有，其他两个样本没有获得边界盒子：

![sift_a_3](/images/ps3/sift_a_3.png)

![sift_d_4](/images/ps3/sift_d_4.png)


# 3. HOG

![hog](/images/ps3/hog.png)

