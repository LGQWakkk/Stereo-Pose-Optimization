// 20250331 双目相机位姿非线性优化
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <fstream>
#include <sstream>
#include "projection_factor.h"  // 视觉残差定义
#include "local.h"
#include <opencv2/opencv.hpp>

#define DEFAULT_FRAME_COUNT 400         //读取多少帧进行优化
int frame_count = DEFAULT_FRAME_COUNT;  // 优化帧数
int frame_skip = 4;                     // 每隔多少帧读取一个帧 用于降采样 提高距离
std::string cam1_data_file = "../data/cam1_data.txt"; // 相机1观测数据
std::string cam2_data_file = "../data/cam2_data.txt"; // 相机2观测数据

// 优化全局变量
double MarkerPose[DEFAULT_FRAME_COUNT][7];//所有信标坐标系位姿
double Cam1Pose[7];//相机1转换到世界坐标系 Rwc1 Pwc1
double Cam2Pose[7];//相机2转换到世界坐标系 Rwc2 Pwc2
double ScaleFactor[1];//尺度因子

std::vector<std::vector<Eigen::Vector2d>> cam1_norm_points_buf;//CAM1 窗口帧 归一化信标观测点
std::vector<std::vector<Eigen::Vector2d>> cam2_norm_points_buf;//CAM2 窗口帧 归一化信标观测点
std::vector<double> cam_timestamp_buf;//视觉数据时间戳

// 加载视觉数据
// return true: 读取成功
// return false: 读取失败
bool load_visual_data(
    std::string cam1_data_file, std::string cam2_data_file,
    std::vector<std::vector<Eigen::Vector2d>> &cam1_norm_points_buf,//CAM1归一化坐标系观测储存
    std::vector<std::vector<Eigen::Vector2d>> &cam2_norm_points_buf,//CAM2归一化坐标系观测储存
    std::vector<double> &cam_timestamp_buf //CAM时间戳储存
)
{
    std::ifstream cam1_stream(cam1_data_file);
    std::ifstream cam2_stream(cam2_data_file);
    if (!cam1_stream.is_open()) {
        printf("Error: cannot open file %s\n", cam1_data_file.c_str());
        return false;
    }
    if (!cam2_stream.is_open()) {
        printf("Error: cannot open file %s\n", cam2_data_file.c_str());
        return false;
    }
    std::string cam1_line, cam2_line;   //单行数据
    int read_count = 0;                 //已经读取的帧数量
    int current_skip_count = 0;
    // 开始读取相机数据 读取WINDOW_SIZE个帧
    while (std::getline(cam1_stream, cam1_line) && std::getline(cam2_stream, cam2_line)){
        if(read_count >= frame_count){
            printf("Visual Data Read Done\n");
            break;
        }

        if(current_skip_count <= 0){//当前应该采样
            current_skip_count = frame_skip;//reload
        }else{//跳过此帧
            current_skip_count--;
            continue;
        }

        std::istringstream iss1(cam1_line);
        std::istringstream iss2(cam2_line);
        double timestamp1, timestamp2;
        std::vector<Eigen::Vector2d> cam1_norm_points, cam2_norm_points;//4个归一化坐标信标观测点
        iss1 >> timestamp1;//时间戳us
        iss2 >> timestamp2;
        for (int i = 0; i < 4; i++) {//CAM1 4个观测点数据
            double x, y;
            iss1 >> x >> y;
            cam1_norm_points.push_back(Eigen::Vector2d(x, y));
        }
        for (int i = 0; i < 4; i++) {//CAM2 4个观测点数据
            double x, y;
            iss2 >> x >> y;
            cam2_norm_points.push_back(Eigen::Vector2d(x, y));
        }
        if (timestamp1 != timestamp2){
            printf("Error: timestamp not match\n");
            return false;
        }
        cam1_norm_points_buf.push_back(cam1_norm_points);
        cam2_norm_points_buf.push_back(cam2_norm_points);
        cam_timestamp_buf.push_back(timestamp1 / 1000000.0);//时间戳 这里需要转换为s 原来是us
        read_count++;
    }
    if(read_count < frame_count){
        printf("Data not enough, only read %d frames\n" , read_count);
        return false;
    }
    // 判断时间戳vector大小是否正确 有可能是数据不够了
    if(cam_timestamp_buf.size() != frame_count){
        printf("Error: cam_timestamp_buf size not match\n");
        return false;
    }
    cam1_stream.close();
    cam2_stream.close();
    printf("Visual Data Read Done\n");
    return true;
}

// 对归一化坐标点进行三角化以对Pose提供初始值
// 传入参数: CAM1 CAM2归一化坐标
// 函数直接更新Pose数组
void triangulate(
    const std::vector<std::vector<Eigen::Vector2d>>& cam1_norm_points, 
    const std::vector<std::vector<Eigen::Vector2d>>& cam2_norm_points,
    const cv::Mat& cam1_proj,//CAM1投影矩阵
    const cv::Mat& cam2_proj //CAM2投影矩阵
)
{
    for(int frame = 0; frame < frame_count; frame++){
        std::vector<Eigen::Vector3d> marker_3d_point;//单帧4个信标三维点
        for(int j = 0; j < 4; j++){//遍历信标点 1 2 3 4
            // 归一化坐标提取
            Eigen::Vector2d cam1_norm(cam1_norm_points_buf[frame][j].x(), cam1_norm_points_buf[frame][j].y());
            Eigen::Vector2d cam2_norm(cam2_norm_points_buf[frame][j].x(), cam2_norm_points_buf[frame][j].y());
            cv::Mat cam1_points(2, 1, CV_64F);//2x1
            cv::Mat cam2_points(2, 1, CV_64F);//2x1
            cam1_points.at<double>(0, 0) = cam1_norm_points_buf[frame][j].x();
            cam1_points.at<double>(1, 0) = cam1_norm_points_buf[frame][j].y();
            cam2_points.at<double>(0, 0) = cam2_norm_points_buf[frame][j].x();
            cam2_points.at<double>(1, 0) = cam2_norm_points_buf[frame][j].y();
            // 三角化
            cv::Mat point4d;
            cv::triangulatePoints(cam1_proj, cam2_proj, cam1_points, cam2_points, point4d);
            if (point4d.at<double>(3, 0) != 0) {//有效三维点
                Eigen::Vector3d point_tri;//信标三维点
                point_tri << point4d.at<double>(0, 0) / point4d.at<double>(3, 0),
                             point4d.at<double>(1, 0) / point4d.at<double>(3, 0),
                             point4d.at<double>(2, 0) / point4d.at<double>(3, 0);
                marker_3d_point.push_back(point_tri);//存入单个信标三维点
            }
        }
        // 根据信标三维点计算信标坐标系 1 2 3 4
        Eigen::Vector3d marker_origin = marker_3d_point[2];//使用信标3作为信标坐标系原点(暂时)
        Eigen::Vector3d x_axis = marker_3d_point[3] - marker_origin;//信标4-信标3作为信标坐标系X轴正方向
        x_axis.normalize();
        Eigen::Vector3d y_axis = marker_3d_point[1] - marker_3d_point[0];//信标2-信标1作为信标坐标系Y轴正方向
        y_axis.normalize();
        Eigen::Vector3d z_axis = x_axis.cross(y_axis);//根据X轴和Y轴计算Z轴
        z_axis.normalize();
        y_axis = z_axis.cross(x_axis);// 根据Z轴和X轴重新计算Y轴
        y_axis.normalize();
        Eigen::Matrix3d marker_matrix;
        marker_matrix << x_axis, y_axis, z_axis;//世界坐标系转换到信标坐标系
        // 计算视觉位姿
        Eigen::Quaterniond marker_quat(marker_matrix.transpose());// 构建四元数(从信标坐标系到世界坐标系)
        MarkerPose[frame][0] = marker_origin.x();
        MarkerPose[frame][1] = marker_origin.y();
        MarkerPose[frame][2] = marker_origin.z();
        MarkerPose[frame][3] = marker_quat.x();
        MarkerPose[frame][4] = marker_quat.y();
        MarkerPose[frame][5] = marker_quat.z();
        MarkerPose[frame][6] = marker_quat.w();
    }
    printf("Triangulation Done\n");
}

// 执行相机以及信标位姿优化
void optimization(Eigen::Matrix<double, 4, 3> &marker_points_m)
{
	printf("Begin Optimization\n");
	ceres::Problem problem;
    ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
    problem.AddParameterBlock(ScaleFactor, 1);//添加尺度因子参数块
	problem.AddParameterBlock(Cam1Pose, 7, local_parameterization);//添加相机1位姿参数块
	problem.AddParameterBlock(Cam2Pose, 7, local_parameterization);//添加相机2位姿参数块
    
	problem.SetParameterBlockConstant(Cam1Pose);//可以选择固定CAM1位姿

    for(int i = 0; i < frame_count; i++)
    {
        problem.AddParameterBlock(MarkerPose[i], 7, local_parameterization);//添加Pose参数
        for(int j = 0; j < 4; j++)
        {
            Eigen::Vector3d marker_point_m = marker_points_m.row(j); //信标在信标坐标系下的坐标
            Eigen::Vector3d marker_point_c1(cam1_norm_points_buf[i][j].x(), cam1_norm_points_buf[i][j].y(), 1.0);
            Eigen::Vector3d marker_point_c2(cam2_norm_points_buf[i][j].x(), cam2_norm_points_buf[i][j].y(), 1.0);
            ProjectionFactor *proj_factor_marker_cam1 = new ProjectionFactor(marker_point_m, marker_point_c1);//mb mp
            ProjectionFactor *proj_factor_marker_cam2 = new ProjectionFactor(marker_point_m, marker_point_c2);//mb mp
            problem.AddResidualBlock(proj_factor_marker_cam1, NULL, Cam1Pose, MarkerPose[i], ScaleFactor);
            problem.AddResidualBlock(proj_factor_marker_cam2, NULL, Cam2Pose, MarkerPose[i], ScaleFactor);
        }
    }
    printf("Begin Solve\n");
    ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_QR;
	options.num_threads = 8;
	options.max_num_iterations = 50;
	// options.max_solver_time_in_seconds = 1.0;  //s
	// options.logging_type = ceres::SILENT;
	// options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
    printf("Done\n");
}

int main()
{
	printf("20250331 Camera Pose Optimization\n");

	// 构建信标点信标坐标系坐标
	Eigen::Matrix<double, 4, 3> marker_points_m;//信标点在信标坐标系下的坐标
	marker_points_m <<
						-0.045,-0.045,  0.000,//marker1
						-0.045, 0.045,  0.000,//marker2
						0.005,  0.000,  0.000,//marker3
						0.045,  0.000,  0.000;//marker4

	// 构建相机矩阵(仅仅作为优化初始值以及三角化参考)
	Eigen::Matrix3d Rc1w = Eigen::Matrix3d::Identity();//CAM1作为参考坐标系(世界坐标系)
	Eigen::Matrix3d Rwc1 = Rc1w.transpose();
	Eigen::Vector3d Pc1w = Eigen::Vector3d::Zero();
	Eigen::Vector3d Pwc1 = -Rwc1 * Pc1w;
	Eigen::Quaterniond Qc1w(Rc1w);
	Eigen::Quaterniond Qwc1(Rwc1);

	Eigen::Matrix3d Rc2w;//世界坐标系转换为相机2坐标系
	Rc2w << 0.999822, -0.009144,  0.016492,
			 0.001514,  0.910672,  0.413128,
			 -0.018796, -0.41303,  0.910523;

	Eigen::Matrix3d Rwc2 = Rc2w.transpose();
	Eigen::Vector3d Pc2w; Pc2w << 0.007412, -0.978053, 0.208224;

	Eigen::Vector3d Pwc2 = -Rwc2 * Pc2w;
	Eigen::Quaterniond Qc2w(Rc2w);
	Eigen::Quaterniond Qwc2(Rwc2);

	// 构建OpenCV相机投影矩阵 用于三角化
	cv::Mat cam1_proj = cv::Mat::zeros(3, 4, CV_64F);
	cv::Mat cam2_proj = cv::Mat::zeros(3, 4, CV_64F);
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			cam1_proj.at<double>(i, j) = Rc1w(i, j);
			cam2_proj.at<double>(i, j) = Rc2w(i, j);
		}
		cam1_proj.at<double>(i, 3) = Pc1w(i);
		cam2_proj.at<double>(i, 3) = Pc2w(i);
	}

	// 将当前相机位姿输入到参数块中
	Cam1Pose[0] = Pwc1.x();
	Cam1Pose[1] = Pwc1.y();
	Cam1Pose[2] = Pwc1.z();
	Cam1Pose[3] = Qwc1.x();
	Cam1Pose[4] = Qwc1.y();
	Cam1Pose[5] = Qwc1.z();
	Cam1Pose[6] = Qwc1.w();

	Cam2Pose[0] = Pwc2.x();
	Cam2Pose[1] = Pwc2.y();
	Cam2Pose[2] = Pwc2.z();
	Cam2Pose[3] = Qwc2.x();
	Cam2Pose[4] = Qwc2.y();
	Cam2Pose[5] = Qwc2.z();
	Cam2Pose[6] = Qwc2.w();

	// 初始化尺度因子
	ScaleFactor[0] = 1.0;

	// 加载视觉数据
	printf("Load Visual Data\n");
    bool ret = load_visual_data(cam1_data_file, cam2_data_file, cam1_norm_points_buf, cam2_norm_points_buf, cam_timestamp_buf);
    if(!ret){
        printf("Error: load visual data failed\n");
        return -1;
    }
	// 三角化获取信标坐标系先验存入参数块
    printf("Triangulation\n");
    triangulate(cam1_norm_points_buf, cam2_norm_points_buf, cam1_proj, cam2_proj);

	// 优化
	optimization(marker_points_m);

    printf("Optimization Done\n");
	// 尺度因子
	printf("Scale Factor: %f\n", ScaleFactor[0]);
	// 相机1位姿
	// printf("Cam1 Pose: px: %f, py: %f, pz: %f, qx: %f, qy: %f, qz: %f, qw: %f\n", Cam1Pose[0], Cam1Pose[1], Cam1Pose[2], Cam1Pose[3], Cam1Pose[4], Cam1Pose[5], Cam1Pose[6]);
	// 相机2位姿
	// printf("Cam2 Pose: px: %f, py: %f, pz: %f, qx: %f, qy: %f, qz: %f, qw: %f\n", Cam2Pose[0], Cam2Pose[1], Cam2Pose[2], Cam2Pose[3], Cam2Pose[4], Cam2Pose[5], Cam2Pose[6]);
	// 相机1转换到世界坐标系旋转矩阵
	Eigen::Matrix3d Rwc1_opt = Eigen::Quaterniond(Cam1Pose[6], Cam1Pose[3], Cam1Pose[4], Cam1Pose[5]).toRotationMatrix();
	// std::cout << "Rwc1_opt: " << std::endl << Rwc1_opt << std::endl;
	// 相机2转换到世界坐标系旋转矩阵
	Eigen::Matrix3d Rwc2_opt = Eigen::Quaterniond(Cam2Pose[6], Cam2Pose[3], Cam2Pose[4], Cam2Pose[5]).toRotationMatrix();
	// std::cout << "Rwc2_opt: " << std::endl << Rwc2_opt << std::endl;
	// 相机1转换到世界坐标系平移向量
	Eigen::Vector3d Pwc1_opt = Eigen::Vector3d(Cam1Pose[0], Cam1Pose[1], Cam1Pose[2]);	
	// std::cout << "Pwc1_opt: " << Pwc1_opt.transpose() << std::endl;
	// 相机2转换到世界坐标系平移向量
	Eigen::Vector3d Pwc2_opt = Eigen::Vector3d(Cam2Pose[0], Cam2Pose[1], Cam2Pose[2]);
	// std::cout << "Pwc2_opt: " << Pwc2_opt.transpose() << std::endl;

    // 实际使用的是世界坐标系转换到相机的矩阵
    Eigen::Matrix3d Rc1w_opt = Rwc1_opt.transpose();
    Eigen::Matrix3d Rc2w_opt = Rwc2_opt.transpose();
    Eigen::Vector3d Pc1w_opt = -Rc1w_opt * Pwc1_opt;
    Eigen::Vector3d Pc2w_opt = -Rc2w_opt * Pwc2_opt;
    std::cout << "Rc1w_opt: " << std::endl << Rc1w_opt << std::endl;
    std::cout << "Pc1w_opt: " << Pc1w_opt.transpose() << std::endl;
    std::cout << "Rc2w_opt: " << std::endl << Rc2w_opt << std::endl;
    std::cout << "Pc2w_opt: " << Pc2w_opt.transpose() << std::endl;

    // 输出相机校准数据到文件
    printf("Output CAMERA Pose\n");
    std::ofstream ofs("cam_pose_opt.txt");
    // 输出CAM1 Pose
    for(int i=0;i<7;i++){
        ofs << Cam1Pose[i] << " ";
    }
    ofs << std::endl;
    // 输出CAM2 Pose
    for(int i=0;i<7;i++){
        ofs << Cam2Pose[i] << " ";
    }
    ofs << std::endl;
    // 输出尺度因子
    ofs << ScaleFactor[0] << std::endl;
    ofs.close();

    // 输出信标坐标系位置到文件
    printf("Output MARKER Pose\n");
    std::ofstream ofs_marker("marker_pose_opt.txt");
    // 输出Pose
    for(int i=0;i<frame_count;i++){
        for(int j=0;j<7;j++){
            ofs_marker << MarkerPose[i][j] << " ";
        }
        ofs_marker << std::endl;
    }
    ofs_marker.close();

	return 0;
}
