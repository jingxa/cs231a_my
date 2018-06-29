# 解题过程

未完待续

- [part 1](https://jingxa.github.io/2018/06/23/CS231A-Homework-2-1/)
- [part 2](https://jingxa.github.io/2018/06/25/CS231A-Homework-2-2/)
- [part 3](https://jingxa.github.io/2018/06/25/CS231A-Homework-2-3/)
- [part 4](https://jingxa.github.io/2018/06/29/CS231A-Homework-2-4/)


# 我的答案

## part 1

```
Set: data/set1
--------------------------------------------------------------------------------
Fundamental Matrix from LLS  8-point algorithm:
 [[ 1.55218081e-06 -8.18161523e-06 -1.50440111e-03]
 [-5.86997052e-06 -3.02892219e-07 -1.13607605e-02]
 [-3.52312036e-03  1.41453881e-02  9.99828068e-01]]
Distance to lines in image 1 for LLS: 28.025662937533877
Distance to lines in image 2 for LLS: 25.162875800036915
p'^T F p = 0.03156399064220228
Fundamental Matrix from normalized 8-point algorithm:
 [[ 5.93261511e-07 -5.08492255e-06  8.76427688e-05]
 [-4.66834735e-06 -3.20108624e-07 -6.12207138e-03]
 [-7.74714403e-04  8.42028676e-03  1.25311400e-01]]
Distance to lines in image 1 for normalized: 0.9431072572196602
Distance to lines in image 2 for normalized: 0.8719800541568359
--------------------------------------------------------------------------------
Set: data/set2
--------------------------------------------------------------------------------
Fundamental Matrix from LLS  8-point algorithm:
 [[-5.63087200e-06  2.74976583e-05 -6.42650411e-03]
 [-2.77622828e-05 -6.74748522e-06  1.52182033e-02]
 [ 1.07623595e-02 -1.22519240e-02 -9.99730547e-01]]
Distance to lines in image 1 for LLS: 9.701438829435915
Distance to lines in image 2 for LLS: 14.568227190498229
p'^T F p = 0.03149037056281445
Fundamental Matrix from normalized 8-point algorithm:
 [[-1.53880961e-07  2.46528633e-06 -1.57563630e-04]
 [ 3.50323566e-06  3.08159735e-07  6.82243058e-03]
 [ 2.42265054e-04 -8.27925885e-03 -4.08002117e-03]]
Distance to lines in image 1 for normalized: 0.8955997529976532
Distance to lines in image 2 for normalized: 0.8959928005846117
```
- 极线图：
<<<<<<< HEAD
![](/ps2_code/result/part_1.png)
=======
![](/result/part_1.png)


---
## Part 2
- 结果为：

```
e1 [-1.30071143e+03 -1.42448272e+02  1.00000000e+00]
e2 [1.65412463e+03 4.53021078e+01 1.00000000e+00]
H1:
 [[-1.20006316e+01 -4.15501447e+00 -1.23476881e+02]
 [ 1.41006481e+00 -1.48704147e+01 -2.84177469e+02]
 [-9.21889298e-03 -2.19184511e-03 -1.23033440e+01]]
H2:
 [[ 8.09798131e-01 -1.22036874e-01  7.99331183e+01]
 [-3.00186699e-02  1.01581538e+00  3.63604348e+00]
 [-6.99360915e-04  1.05393946e-04  1.15205554e+00]]
```
>>>>>>> finish ps2

调整过后的图片为：

![](https://upload-images.jianshu.io/upload_images/5361608-d76d7018232d0b04.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


## Part 3
- 结果为：
```
(4, 4) (4,) (37, 37)
[959.5852216  540.47613178 184.43174791  27.9151956 ]
(4, 4) (4,) (18, 18)
[264.54396508 210.06072009   7.21921783   5.12857709]
```
图对比：

![](https://upload-images.jianshu.io/upload_images/5361608-88e8a61ad5260eab.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)





## part 4

```
Part A: Check your matrices against the example R,T
--------------------------------------------------------------------------------
Example RT:
 [[ 0.9736 -0.0988 -0.2056  0.9994]
 [ 0.1019  0.9948  0.0045 -0.0089]
 [ 0.2041 -0.0254  0.9786  0.0331]]
Estimated RT:
 [[[ 0.98305251 -0.11787055 -0.14040758  0.99941228]
  [-0.11925737 -0.99286228 -0.00147453 -0.00886961]
  [-0.13923158  0.01819418 -0.99009269  0.03311219]]

 [[ 0.98305251 -0.11787055 -0.14040758 -0.99941228]
  [-0.11925737 -0.99286228 -0.00147453  0.00886961]
  [-0.13923158  0.01819418 -0.99009269 -0.03311219]]

 [[ 0.97364135 -0.09878708 -0.20558119  0.99941228]
  [ 0.10189204  0.99478508  0.00454512 -0.00886961]
  [ 0.2040601  -0.02537241  0.97862951  0.03311219]]

 [[ 0.97364135 -0.09878708 -0.20558119 -0.99941228]
  [ 0.10189204  0.99478508  0.00454512  0.00886961]
  [ 0.2040601  -0.02537241  0.97862951 -0.03311219]]]
--------------------------------------------------------------------------------
Part B: Check that the difference from expected point 
is near zero
--------------------------------------------------------------------------------
Difference:  0.0029243053036863698
--------------------------------------------------------------------------------
Part C: Check that the difference from expected error/Jacobian 
is near zero
--------------------------------------------------------------------------------
Error Difference:  8.301299988565727e-07
Jacobian Difference:  1.817115702351657e-08
--------------------------------------------------------------------------------
Part D: Check that the reprojection error from nonlinear method
is lower than linear method
--------------------------------------------------------------------------------
Linear method error: 98.73542356894183
Nonlinear method error: 95.59481784846034
--------------------------------------------------------------------------------
Part E: Check your matrix against the example R,T
--------------------------------------------------------------------------------
Example RT:
 [[ 0.9736 -0.0988 -0.2056  0.9994]
 [ 0.1019  0.9948  0.0045 -0.0089]
 [ 0.2041 -0.0254  0.9786  0.0331]]
Estimated RT:
 [[ 0.97364135 -0.09878708 -0.20558119  0.99941228]
 [ 0.10189204  0.99478508  0.00454512 -0.00886961]
 [ 0.2040601  -0.02537241  0.97862951  0.03311219]]
--------------------------------------------------------------------------------
Part F: Run the entire SFM pipeline
--------------------------------------------------------------------------------
```

![](https://upload-images.jianshu.io/upload_images/5361608-cd68806911f3d3b9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

