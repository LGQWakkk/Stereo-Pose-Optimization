## 双目相对位姿非线性优化

### Theory

**符号表示**

$\mathbf{R}_c^w$​：相机坐标系转换到世界坐标系旋转矩阵

$\mathbf{R}_m^w$​：信标坐标系转换到世界坐标系旋转矩阵

$\mathbf{p}_c^w$​：相机坐标系原点在世界坐标系位置

$\mathbf{p}_m^w$​：坐标信标系原点在世界坐标系位置

$scale$：尺度因子——相机三角化距离乘以此因子转换为米

$\mathbf{m}_c$：信标点在相机坐标系中坐标

$\mathbf{m}_b$：信标点在信标坐标系中坐标

**优化流程**

本程序需要相机相对位姿的初始估计，以及所有信标点在两个相机归一化平面的坐标值(存放在data文件夹中)，还有信标点在信标坐标系中的坐标。

首先读取归一化平面坐标，根据初始位姿进行三角化，求解得出各帧信标坐标系在相机坐标系的初始估计，之后通过Ceres进行全局的非线性优化，实现对于相机位姿、每一帧信标位姿和尺度因子的优化。

**优化参数块**

$Cam1 Pose[7]$：相机1在世界坐标系的位置和四元数

$Cam2Pose[7]$：相机2在世界坐标系的位置和四元数

$MarkerPose[FrameCount][7]$：信标坐标系在世界坐标系的位置和四元数

$ScaleFactor[1]$：尺度因子

注：本程序中将CAM1看做世界坐标系，两者等价

**关于相对位姿问题**

一般来说，优化是针对CAM1和CAM2相对位姿的，但是程序提供了两个独立的参数块，所以如果不对其中施加限制，则两者在优化之后都会偏移，虽然结果是正确的，但是处理起来比较麻烦。如果想要固定CAM1的位姿，可以解除如下代码的注释：

```c++
//problem.SetParameterBlockConstant(Cam1Pose);//可以选择固定CAM1位姿
```

**残差计算以及雅可比矩阵推导**

信标点转换到相机坐标系：
$$m_c={\mathrm {R}_c^w}^T(\mathrm {R}_m^w\frac{m_b}{scale}+\mathrm {p}_m^w-\mathrm {p}_c^w)$$
Visual Jacobian：
$$
\frac{\partial m_c}{\partial \mathrm{q}_m^w} =-{\mathrm{R}_c^w}^T{\mathrm{R}_m^w}(\frac{m_b}{scale})^{\wedge} 
\frac{\partial m_c}{\partial \mathrm{p}_m^w} = {\mathrm{R}_c^w}^T 
\frac{\partial m_c}{\partial \mathrm{q}_c^w} = \left [ {\mathrm {R}_c^w}^T(\mathrm {R}_m^w\frac{m_b}{scale}+\mathrm {p}_m^w-\mathrm {p}_c^w) \right ]^{\wedge }
\frac{\partial m_c}{\partial \mathrm{p}_c^w} = -{\mathrm{R}_c^w}^T
\frac{\partial m_c}{\partial scale} = -\frac{1}{scale^2}{\mathrm{R}_c^w}^T{\mathrm{R}_m^w}m_b
$$
二维残差对$m_c$求导的部分和大多数视觉里程计一致，这里不再赘述
残差具体定义可参考 projection_factor.h

**参数配置** main.cpp

```c++
#define DEFAULT_FRAME_COUNT 400         // 读取多少帧进行优化
int frame_count = DEFAULT_FRAME_COUNT;  // 优化帧数
int frame_skip = 4;                     // 每隔多少帧读取一个帧 用于降采样 提高距离
std::string cam1_data_file = "../data/cam1_data.txt"; // 相机1观测数据
std::string cam2_data_file = "../data/cam2_data.txt"; // 相机2观测数据
```

data文件夹下含有相机观测数据

相机观测数据格式说明：（以本次4个信标点情况为例）

timestamp px1 py1 px2 py2 px3 py3 px4 py4

共9列数据，其中timestamp为时间戳，单位us，px py为相机归一化平面坐标(Z坐标为1)，注意这里的信标观测序号应该和如下的信标点序号相对应：

```c++
	// 构建信标点信标坐标系坐标
	Eigen::Matrix<double, 4, 3> marker_points_m;//信标点在信标坐标系下的坐标
	marker_points_m <<
			-0.045,-0.045,  0.000,//marker1
			-0.045, 0.045,  0.000,//marker2
			0.005,  0.000,  0.000,//marker3
			0.045,  0.000,  0.000;//marker4
```

**信标说明**

具体工程详见：[LGQWakkk/MARKER: 四点式手持主动红外信标 视觉惯性采集设备](https://github.com/LGQWakkk/MARKER)

![](output\marker.png)

### 优化结果

优化帧数：400

优化时间：261秒 (8线程 但其实是单核)

初始残差：6.799694e+00

终止残差：2.924840e-03

```latex
(ACADOS) wakkk@wakkk-virtual-machine:~/CeresProjects/StereoPoseOptimization/Ceres/build$ ./main
20250331 Camera Pose Optimization
Load Visual Data
Visual Data Read Done
Visual Data Read Done
Triangulation
Triangulation Done
Begin Optimization
Begin Solve

Solver Summary (v 2.0.0-eigen-(3.4.0)-lapack-eigensparse-no_openmp)

                                     Original                  Reduced
Parameter blocks                          403                      402
Parameters                               2815                     2808
Effective parameters                     2413                     2407
Residual blocks                          3200                     3200
Residuals                                6400                     6400

Minimizer                        TRUST_REGION

Dense linear algebra library            EIGEN
Trust region strategy     LEVENBERG_MARQUARDT

                                        Given                     Used
Linear solver                        DENSE_QR                 DENSE_QR
Threads                                     8                        8
Linear solver ordering              AUTOMATIC                      402

Cost:
Initial                          6.799694e+00
Final                            2.924840e-03
Change                           6.796770e+00

Minimizer iterations                       16
Successful steps                           11
Unsuccessful steps                          5

Time (in seconds):
Preprocessor                         0.238685

  Residual only evaluation           0.123134 (16)
  Jacobian & residual evaluation     0.626561 (11)
  Linear solver                    259.126028 (16)
Minimizer                          260.670821

Postprocessor                        0.000921
Total                              260.910428

Termination:                      CONVERGENCE (Function tolerance reached. |cost_change|/cost: 1.560248e-07 <= 1.000000e-06)

Done
Optimization Done
Scale Factor: 0.418336
Rc1w_opt: 
1 0 0
0 1 0
0 0 1
Pc1w_opt: 0 0 0
Rc2w_opt: 
  0.999688    -0.0172184   0.0180863
  0.00810031   0.908703    0.417366
 -0.0236215   -0.417089    0.908559
Pc2w_opt: 0.0145059 -0.978387  0.209445
Output CAMERA Pose
Output MARKER Pose
```

信标坐标系在CAM1中优化结果：

![](output\marker_frame.png)

信标坐标系原点：

![](output\marker_origin.png)

优化之前相对位姿：

```latex
Rc2w: 
	0.999822    -0.009144    0.016492
	0.001514     0.910672  	 0.413128
	-0.018796   -0.41303     0.910523
Pc2w: 
	0.007412    -0.978053    0.208224
```

优化之后相对位姿：

```latex
Rc2w_opt: 
    0.999688    -0.0172184   0.0180863
    0.00810031   0.908703    0.417366
   -0.0236215   -0.417089    0.908559
Pc2w_opt: 
    0.0145059   -0.978387    0.209445
```

