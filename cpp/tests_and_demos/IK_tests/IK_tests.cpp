/**
 * Use CSV compiler in matlab tests folder to generate sp_x.csv for each subproblem
 * Otherwise, this will not work.
 *
 * timing tests of all subproblems
 *
 */

#include <iostream>
#include <chrono>
#include <string>
#include <eigen3/Eigen/Dense>

#include "Spherical_IK.h"
#include "../read_csv.h"

#define ERROR_PASS_EPSILON 1e-9
#define BATCH_SIZE 1

bool ik_test_SPHERICAL_1_2_P();
bool ik_test_SPHERICAL_2_3_P();
bool ik_test_SPHERICAL_1_3_P();
bool ik_test_SPHERICAL();
bool ik_test_SPHERICAL_1_2_I();
bool ik_test_SPHERICAL_2_3_I();

bool evaluate_test(const std::string &name_test,
				   const IKS::Robot_Kinematics &robot,
				   const IKS::Homogeneous_T &ee_pose);


int main(int argc, char *argv[])
{ 
	ik_test_SPHERICAL_1_2_P();
	ik_test_SPHERICAL_2_3_P();
	ik_test_SPHERICAL_1_3_P();
	ik_test_SPHERICAL();
	ik_test_SPHERICAL_1_2_I();
	ik_test_SPHERICAL_2_3_I();
	return 0;
}

// spherical wrist, with the remaining second and third axis intersecting
bool ik_test_SPHERICAL_2_3_I()
{
	const Eigen::Vector3d zv(0, 0, 0);
	const Eigen::Vector3d ex(1, 0, 0);
	const Eigen::Vector3d ey(0, 1, 0);
	const Eigen::Vector3d ez(0, 0, 1);

	// Robot configuration for spherical wrist with second and third axis intersecting
	Eigen::Matrix<double, 3, 6> spherical_intersecting_H;
	spherical_intersecting_H << ex, ez, ey, ez, ex, ey;
	Eigen::Matrix<double, 3, 7> spherical_intersecting_P;
	spherical_intersecting_P << ey, -ey+ez, ez, -ey + 2 * ez, zv, zv, 2 * ey;

	IKS::Robot_Kinematics spherical_intersecting(spherical_intersecting_H, spherical_intersecting_P);
	IKS::Homogeneous_T ee_pose_spherical_intersecting = spherical_intersecting.fwdkin({1, 1, 1, 0, 1, 0});

	return evaluate_test("IK spherical wrist - Axis 2 intersecting 3", spherical_intersecting, ee_pose_spherical_intersecting);	
}

// spherical wrist, with the remaining first and second axis intersecting
bool ik_test_SPHERICAL_1_2_I()
{
	const Eigen::Vector3d zv(0, 0, 0);
	const Eigen::Vector3d ex(1, 0, 0);
	const Eigen::Vector3d ey(0, 1, 0);
	const Eigen::Vector3d ez(0, 0, 1);

	// Robot configuration for spherical wrist with first and second axis intersecting
	Eigen::Matrix<double, 3, 6> spherical_intersecting_H;
	spherical_intersecting_H << ez, ey, ex, ez, ex, ey;
	Eigen::Matrix<double, 3, 7> spherical_intersecting_P;
	spherical_intersecting_P << ez, zv, ez + ey, -ey + 2 * ez, zv, zv, 2 * ey;

	/*  Real-Life example:
		Partial IK: KukaR800FixedQ3
		H << ez, ex, 0.5*ex-0.8660254037844387*ey, ez,  0.5*ex+0.8660254037844387*ey, ez;
		P << 0.33999999999999997*ez, zv, 0.4* ez, 0.4* ez, zv, zv, 0.126*ez;
	*/

	IKS::Robot_Kinematics spherical_intersecting(spherical_intersecting_H, spherical_intersecting_P);
	IKS::Homogeneous_T ee_pose_spherical_intersecting = spherical_intersecting.fwdkin({1, 1, 1, 0, 1, 0});

	return evaluate_test("IK spherical wrist - Axis 1 intersecting 2", spherical_intersecting, ee_pose_spherical_intersecting);	
}

// spherical wrist, with the remaining first and third axis parallel (Solvable by SP5)
bool ik_test_SPHERICAL_1_3_P()
{
	const Eigen::Vector3d zv(0, 0, 0);
	const Eigen::Vector3d ex(1, 0, 0);
	const Eigen::Vector3d ey(0, 1, 0);
	const Eigen::Vector3d ez(0, 0, 1);

	// Using modified version of Irb6640 where first axis switches place with second and third
	Eigen::Matrix<double, 3, 6> Irb6640_mod_H;
	Irb6640_mod_H << ey, ez, ey, ex, ey, ex;
	Eigen::Matrix<double, 3, 7> Irb6640_mod_P;
	Irb6640_mod_P << zv, 0.32 * ex + 0.78 * ez, 1.075 * ez, 1.1425 * ex + 0.2 * ez, zv, zv, 0.2 * ex;

	IKS::Robot_Kinematics Irb6640_mod(Irb6640_mod_H, Irb6640_mod_P);
	IKS::Homogeneous_T ee_pose_Irb6640_mod = Irb6640_mod.fwdkin({1, 1, 1, 0, 1, 0});

	return evaluate_test("IK spherical wrist - Axis 1||3", Irb6640_mod, ee_pose_Irb6640_mod);	
}

// spherical wrist, with the remaining first and second axis parallel
bool ik_test_SPHERICAL_1_2_P()
{
	const Eigen::Vector3d zv(0, 0, 0);
	const Eigen::Vector3d ex(1, 0, 0);
	const Eigen::Vector3d ey(0, 1, 0);
	const Eigen::Vector3d ez(0, 0, 1);

	// Using modified version of Irb6640 where first axis switches place with second and third
	Eigen::Matrix<double, 3, 6> Irb6640_mod_H;
	Irb6640_mod_H << ey, ey, ez, ex, ey, ex;
	Eigen::Matrix<double, 3, 7> Irb6640_mod_P;
	Irb6640_mod_P << zv, 0.32 * ex + 0.78 * ez, 1.075 * ez, 1.1425 * ex + 0.2 * ez, zv, zv, 0.2 * ex;

	IKS::Robot_Kinematics Irb6640_mod(Irb6640_mod_H, Irb6640_mod_P);
	IKS::Homogeneous_T ee_pose_Irb6640_mod = Irb6640_mod.fwdkin({1, 1, 1, 0, 1, 0});

	return evaluate_test("IK spherical wrist - Axis 2||3", Irb6640_mod, ee_pose_Irb6640_mod);	
}

// spherical wrist, with the remaining second and third axis parallel
bool ik_test_SPHERICAL_2_3_P()
{
	const Eigen::Vector3d zv(0, 0, 0);
	const Eigen::Vector3d ex(1, 0, 0);
	const Eigen::Vector3d ey(0, 1, 0);
	const Eigen::Vector3d ez(0, 0, 1);

	Eigen::Matrix<double, 3, 6> Irb6640_H;
	Irb6640_H << ez, ey, ey, ex, ey, ex;
	Eigen::Matrix<double, 3, 7> Irb6640_P;
	Irb6640_P << zv, 0.32 * ex + 0.78 * ez, 1.075 * ez, 1.1425 * ex + 0.2 * ez, zv, zv, 0.2 * ex;

	IKS::Robot_Kinematics Irb6640(Irb6640_H, Irb6640_P);
	IKS::Homogeneous_T ee_pose_Irb6640 = Irb6640.fwdkin({1, 1, 1, 0, 1, 0});

	return evaluate_test("IK spherical wrist - Axis 2||3", Irb6640, ee_pose_Irb6640);
}

bool ik_test_SPHERICAL()
{
	const Eigen::Vector3d zv(0, 0, 0);
	const Eigen::Vector3d ex(1, 0, 0);
	const Eigen::Vector3d ey(0, 1, 0);
	const Eigen::Vector3d ez(0, 0, 1);

	Eigen::Matrix<double, 3, 6> Spherical_Bot_H;
	Spherical_Bot_H << ey, ez, ey, ex, ey, ex;
	Eigen::Matrix<double, 3, 7> Spherical_Bot_P;
	Spherical_Bot_P << zv, ez + ex, ez + ex, ez + ex, zv, zv, ex;

	IKS::Robot_Kinematics Spherical_Bot(Spherical_Bot_H, Spherical_Bot_P);
	IKS::Homogeneous_T ee_pose_Spherical_Bot = Spherical_Bot.fwdkin({1, 1, 1, 0, 0, 0});

	return evaluate_test("IK spherical wrist", Spherical_Bot, ee_pose_Spherical_Bot);
}

bool evaluate_test(const std::string &name_test, const IKS::Robot_Kinematics &robot, const IKS::Homogeneous_T &ee_pose)
{
	IKS::IK_Solution solution;
	const auto start = std::chrono::steady_clock::now();
	for (unsigned i = 0; i < BATCH_SIZE; ++i)
	{
		solution = robot.calculate_IK(ee_pose);
	}

	const auto end = std::chrono::steady_clock::now();
	unsigned long time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

	double sum_error = 0;
	double max_error = 0;
	for (const auto &solution : solution.Q)
	{
		IKS::Homogeneous_T result = robot.fwdkin(solution);

		double error = (result - ee_pose).norm();
		max_error = max_error < error ? error : max_error;
		sum_error += error;
	}

	const double avg_error = sum_error / solution.Q.size();
	const bool is_passed{std::fabs(max_error) < ERROR_PASS_EPSILON &&
						 std::fabs(avg_error) < ERROR_PASS_EPSILON};
	std::cout << "Test [" << name_test << "]: ";
	if (is_passed)
	{
		std::cout << "[PASS]" << std::endl;
	}
	else
	{
		std::cout << "[FAIL]" << std::endl;
	}
	std::cout << "\tAverage error: " << avg_error << std::endl;
	std::cout << "\tMaximum error: " << max_error << std::endl;
	std::cout << "===== \n Average solution time (nanoseconds): " << time / BATCH_SIZE << std::endl;

	return is_passed;
}