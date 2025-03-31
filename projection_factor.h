#pragma once
// 20250331 双目相机相对位姿优化
// 通过多个信标点以及多个测量 
// 实现对于双目相机相对位姿的优化以及尺度因子的测量

#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>
#include "utility.h"

// 参数块:
// 信标坐标系Pose(7) 相机坐标系Pose(7) 尺度因子(1)
// 信标坐标系: 信标坐标系转换到世界坐标系
// 相机坐标系：相机坐标系转换到世界坐标系
class ProjectionFactor : public ceres::SizedCostFunction<2, 7, 7, 1>
{
    public:
    ProjectionFactor() = delete;
    ProjectionFactor(
        const Eigen::Vector3d &_mb,    //信标点在IMU坐标系中的坐标
        const Eigen::Vector3d &_mp     //信标点实际观测值(归一化平面坐标)
    ) : mb(_mb), mp(_mp){}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        // 参数块1 相机坐标系转换为世界坐标系
        Eigen::Vector3d Pwc(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qwc(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
        // 参数块2 信标坐标系转换为世界坐标系
        Eigen::Vector3d Pwm(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Quaterniond Qwm(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);
        // 参数块3 尺度因子
        double scale = parameters[2][0];//尺度因子

        Eigen::Vector3d mw = Qwm * mb / scale + Pwm;        //世界坐标系中信标点坐标
        Eigen::Vector3d mc = Qwc.inverse() * (mw - Pwc);    //相机坐标系中信标点坐标
        double depth = mc.z();                              //相机坐标系中信标点深度

        Eigen::Map<Eigen::Vector2d> residual(residuals);
        residual = (mc / depth).head<2>() - mp.head<2>();   //计算残差
        if (jacobians)
        {
            Eigen::Matrix3d Rwc = Qwc.toRotationMatrix();
            Eigen::Matrix3d Rwm = Qwm.toRotationMatrix();
            // 二维残差对mc求导
            Eigen::Matrix<double, 2, 3> reduce(2, 3);
            reduce << 1. / depth, 0, -mc(0) / (depth * depth),
                      0, 1. / depth, -mc(1) / (depth * depth);
            if (jacobians[0])
            {   //mc对相机位姿求导
                Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> _jacobian(jacobians[0]);
                Eigen::Matrix<double, 3, 6> _jaco;
                _jaco.leftCols<3>() = - Rwc.transpose();//对相机位置的导数
                _jaco.rightCols<3>() = Utility::skewSymmetric(mc);//对相机旋转的导数
                _jacobian.leftCols<6>() = reduce * _jaco;//2x3 x 3x6 = 2x6
                _jacobian.rightCols<1>().setZero();//2x7
            }
            if (jacobians[1])
            {   //mc对信标位姿求导
                Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> _jacobian(jacobians[1]);
                Eigen::Matrix<double, 3, 6> _jaco;
                _jaco.leftCols<3>() = Rwc.transpose();//对信标位置的导数
                _jaco.rightCols<3>() = - Rwc.transpose() * Rwm * Utility::skewSymmetric(mb / scale);//对信标旋转的导数
                _jacobian.leftCols<6>() = reduce * _jaco;//2x3 x 3x6 = 2x6
                _jacobian.rightCols<1>().setZero();//2x7
            }
            if(jacobians[2])
            {   //mc对尺度因子求导
                Eigen::Map<Eigen::Matrix<double, 2, 1>> _jacobian(jacobians[2]);
                Eigen::Matrix<double, 3, 1> _jaco = - Rwc.transpose() * Rwm * mb / (scale * scale);//3x1
                _jacobian = reduce * _jaco;//2x3 x 3x1 = 2x1
            }
        }
        return true;
    }
    Eigen::Vector3d mb;//Const 信标点在IMU坐标系中的坐标
    Eigen::Vector3d mp;//Const 信标点实际观测值(归一化平面坐标)
};
