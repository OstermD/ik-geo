#ifndef IK_KINEMATIK_H
#define IK_KINEMATIK_H

#include <vector>
#include <eigen3/Eigen/Dense>

namespace IKS
{
	struct Kin {
		Eigen::Matrix<double, 3, 6> H;
		Eigen::Matrix<double, 3, 7> P;
		Eigen::Matrix<double, 1, 6> joint_type;
	};

	struct Solution {
		Eigen::Matrix<double, 6, Eigen::Dynamic> Q;
		Eigen::Matrix<bool, 6, Eigen::Dynamic> is_LS_vec;
	};

    struct Soln {
		std::vector<std::vector<double>> Q;
		std::vector<bool> is_LS_vec;
	};

}

#endif 