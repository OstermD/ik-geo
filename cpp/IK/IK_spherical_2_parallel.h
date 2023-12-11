//---------------------------------------------------------------//
// Name: IK_spherical_2_parallel.h
// Author: Runbin Chen
// Date: 02/01/2023
//---------------------------------------------------------------//

#ifndef __IK_spherical_2_parallel_h_
#define __IK_spherical_2_parallel_h_

#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "IK_Kinematic.h"

namespace IKS
{
	Solution IK_spherical_2_parallel(const Eigen::Matrix<double, 3, 3>& R_0T, const Eigen::Vector3d& p_0T, const Kin& kin);
}

#endif