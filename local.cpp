#include "local.h"

//更新优化变量使用的
//x:input 原始变量
//delta:增量
//x_plus_delta:更新后的变量

//在这里 x是一个7维度的参数块
//前三个为位移 后四个为四元数
bool PoseLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    Eigen::Map<const Eigen::Vector3d> _p(x);//获取位移
    Eigen::Map<const Eigen::Quaterniond> _q(x + 3);//获取四元数

    Eigen::Map<const Eigen::Vector3d> dp(delta);//位移微小变化
    //注意这里使用的是Vector3d而不是Quaterniond
    //因为这个四元数增量很小 所以使用虚部进行近似
    //而且因为四元数在这个参数块中的顺序是: px py pz x y z w
    //所以从第三个开始提取三个自然就是x y z 四元素的向量部分
    Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));//四元数微小变化

    Eigen::Map<Eigen::Vector3d> p(x_plus_delta);//输出位移结果映射
    Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);//输出四元数结果映射

    p = _p + dp;//对于位置直接加和
    q = (_q * dq).normalized();//对于四元数需要进行乘法并进行归一化

    return true;
}

//应该是不使用这个函数进行雅可比的计算
bool PoseLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
    j.topRows<6>().setIdentity();
    j.bottomRows<1>().setZero();
    return true;
}
