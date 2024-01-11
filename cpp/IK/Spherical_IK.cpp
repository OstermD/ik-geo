#include <eigen3/Eigen/Dense>
#include <vector>

#include "../subproblems/sp.h"
#include "Spherical_IK.h"

namespace IKS
{
    Robot_Kinematics::Robot_Kinematics(const Eigen::Matrix<double, 3, 6> &H, const Eigen::Matrix<double, 3, 7> &P)
        : H(H), P(P)
    {
    }

    IK_Solution Robot_Kinematics::calculate_IK(const Homogeneous_T &ee_position_orientation)
    {
        IK_Solution solution;
        const Eigen::Vector3d p_0t = ee_position_orientation.block<3, 1>(0, 3);
        const Eigen::Matrix3d r_06 = ee_position_orientation.block<3, 3>(0, 0);

        const Eigen::Vector3d p_16 = p_0t - robot_kinematics.P.col(0) - r_06 * robot_kinematics.P.col(6);

        SP5 position_kinematics(-this->P.col(1),
                                p_16,
                                this->P.col(2),
                                this->P.col(3),
                                -this->H.col(0),
                                this->H.col(1),
                                this->H.col(2));
        position_kinematics.solve();
        solution.Q.push_back(position_kinematics.get_theta_1());
        solution.Q.push_back(position_kinematics.get_theta_2());
        solution.Q.push_back(position_kinematics.get_theta_3());
        return solution;
    }
};