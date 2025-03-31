# 绘制标定结果
import numpy as np
import quaternion
import matplotlib.pyplot as plt

cam_pose_file = 'output/cam_pose_opt.txt'
marker_pose_file = 'output/marker_pose_opt.txt'

marker_pose_data = np.loadtxt(marker_pose_file)

with open(cam_pose_file, 'r') as file:
    lines = file.readlines()
cam1_pose = np.array([float(num) for num in lines[0].strip().split()])  # 相机转换到世界
cam2_pose = np.array([float(num) for num in lines[1].strip().split()])  # 相机转换到世界
scale = float(lines[2].strip())  # 尺度因子
print(f"CAM1 Pose: {cam1_pose}")
print(f"CAM2 Pose: {cam2_pose}")
print(f"Scale Factor: {scale}")

# 四元数转换为旋转矩阵
Qwc1 = quaternion.quaternion(cam1_pose[6], cam1_pose[3], cam1_pose[4], cam1_pose[5]) # w x y z
Rwc1 = quaternion.as_rotation_matrix(Qwc1)
Rc1w = Rwc1.T
Pwc1 = np.array([cam1_pose[0], cam1_pose[1], cam1_pose[2]])
Pc1w = - np.dot(Rc1w, Pwc1)

Qwc2 = quaternion.quaternion(cam2_pose[6], cam2_pose[3], cam2_pose[4], cam2_pose[5]) # w x y z
Rwc2 = quaternion.as_rotation_matrix(Qwc2)
Rc2w = Rwc2.T
Pwc2 = np.array([cam2_pose[0], cam2_pose[1], cam2_pose[2]])
Pc2w = - np.dot(Rc2w, Pwc2)

# 绘制信标在CAM1中坐标系位置
marker_count = marker_pose_data.shape[0]
print(f"Marker Frame Count: {marker_count}")
# Pose: x y z qx qy qz qw 为信标坐标系转换到世界坐标系
plot_marker_origin = True  # 是否绘制信标坐标系原点
plot_marker_frame = False   # 是否绘制信标坐标系

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(marker_count):
    marker_pos = marker_pose_data[i, 0:3]
    marker_quat = quaternion.quaternion(marker_pose_data[i, 6], marker_pose_data[i, 3], marker_pose_data[i, 4], marker_pose_data[i, 5])
    marker_rot = quaternion.as_rotation_matrix(marker_quat)  # 信标坐标系转换为CAM1坐标系
    marker_x = np.dot(marker_rot, np.array([1.0, 0.0, 0.0]))
    marker_y = np.dot(marker_rot, np.array([0.0, 1.0, 0.0]))
    marker_z = np.dot(marker_rot, np.array([0.0, 0.0, 1.0]))

    if plot_marker_origin:
        ax.scatter(marker_pos[0], marker_pos[1], marker_pos[2], c='r', marker='o', s=0.5)
    
    if plot_marker_frame:
        ax.quiver(marker_pos[0], marker_pos[1], marker_pos[2], marker_x[0], marker_x[1], marker_x[2], color='r', length=0.1, linewidths=0.2)
        ax.quiver(marker_pos[0], marker_pos[1], marker_pos[2], marker_y[0], marker_y[1], marker_y[2], color='g', length=0.1, linewidths=0.2)
        ax.quiver(marker_pos[0], marker_pos[1], marker_pos[2], marker_z[0], marker_z[1], marker_z[2], color='b', length=0.1, linewidths=0.2)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
