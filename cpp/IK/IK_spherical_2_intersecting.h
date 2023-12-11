//---------------------------------------------------------------//
// Name: IK_spherical_2_intersecting.h
// Author: Amar Maksumic
// Date: 02/03/2022
// Purpose: Port of the IK_spherical_2_intersecting files
//---------------------------------------------------------------//


#ifndef __IK_spherical_2_intersecting_h__
#define __IK_spherical_2_intersecting_h__

#include <vector>
#include <eigen3/Eigen/Dense>
#include "./IK_Kinematic.h"

namespace IKS 
{
	void fwdkin(const Kin& kin, const Soln& soln, 
				Eigen::Matrix<double, 3, 1>& p, 
							Eigen::Matrix<double, 3, 3>& R);

	std::pair<Eigen::Matrix<double, 3, 1>, Eigen::Matrix<double, 3, 3>> fwdkin_py(const Kin& kin, const Solution& soln);

	void setup(Kin& kin, Soln& soln,
						Eigen::Matrix<double, 3, 1>& T, 
						Eigen::Matrix<double, 3, 3>& R);

	void setup_LS(Kin& kin, Soln& soln,
						Eigen::Matrix<double, 3, 1>& T, 
						Eigen::Matrix<double, 3, 3>& R);

	void error();

	void IK_spherical_2_intersecting(const Eigen::Matrix<double, 3, 3>& R_0T, const Eigen::Vector3d& p_0T, const Kin& kin, 
									Eigen::Matrix<double, 6, Eigen::Dynamic>& Q, Eigen::Matrix<double, 5, Eigen::Dynamic>& Q_LS);

	Solution IK_spherical_2_intersecting_py(const Eigen::Matrix<double, 3, 3>& R_0T, const Eigen::Vector3d& p_0T, const Kin& kin);
}
#endif