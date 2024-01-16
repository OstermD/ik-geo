#ifndef SPHERICAL_IK_H
#define SPHERICAL_IK_H

#include <vector>
#include <eigen3/Eigen/Dense>

namespace IKS
{
    using Homogeneous_T = Eigen::Matrix<double, 4, 4>;

    struct IK_Solution
    {
        std::vector<std::vector<double>> Q;
        std::vector<bool> is_LS_vec;
    };

    class Robot_Kinematics
    {
    public:
        Robot_Kinematics(const Eigen::Matrix<double, 3, 6> &H, const Eigen::Matrix<double, 3, 7> &P);
        IK_Solution calculate_IK(const Homogeneous_T &ee_position_orientation) const;
        Homogeneous_T fwdkin(const std::vector<double> &Q) const;

    private:
        Eigen::Matrix<double, 3, 6> H;
        Eigen::Matrix<double, 3, 7> P;
    };
}

#endif