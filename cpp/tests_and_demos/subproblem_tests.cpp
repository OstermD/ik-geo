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

#include "sp.h"
#include "../read_csv.h"
#include <eigen3/Eigen/Dense>

#define PATH_TESTS_SP1 "../../../test_cases/sp_1.csv"
#define PATH_TESTS_SP2 "../../../test_cases/sp_2.csv"
#define PATH_TESTS_SP3 "../../../test_cases/sp_3.csv"
#define PATH_TESTS_SP4 "../../../test_cases/sp_4.csv"
#define PATH_TESTS_SP5 "../../../test_cases/sp_5.csv"
#define PATH_TESTS_SP6 "../../../test_cases/sp_6.csv"

#define ERROR_PASS_EPSILON 1e-9

bool evaluate_test(const std::string &name_test,
					   const double &max_expected,
					   const double &avg_expected,
					   const double &max_actual,
					   const double &avg_actual,
					   const unsigned &num_LS,
					   const unsigned& num_testcases,
					   const double& time_avg);

bool test_SP1(const std::string &csv_path);
bool test_SP2(const std::string &csv_path);
bool test_SP3(const std::string &csv_path);
bool test_SP4(const std::string &csv_path);
bool test_SP5(const std::string &csv_path);
bool test_SP6(const std::string &csv_path);

int main(int argc, char *argv[])
{
	test_SP1(PATH_TESTS_SP1);
	test_SP2(PATH_TESTS_SP2);
	test_SP3(PATH_TESTS_SP3);
	test_SP4(PATH_TESTS_SP4);
	test_SP5(PATH_TESTS_SP5);
	test_SP6(PATH_TESTS_SP6);

	return 0;
}


bool evaluate_test(const std::string &name_test,
					   const double &max_expected,
					   const double &avg_expected,
					   const double &max_actual,
					   const double &avg_actual,
					   const unsigned &num_LS,
					   const unsigned& num_testcases,
					   const double& time_avg)
{
	const bool is_passed{std::fabs(avg_actual - avg_expected) < ERROR_PASS_EPSILON &&
						 std::fabs(max_actual - max_expected) < ERROR_PASS_EPSILON &&
						 num_LS == 0};
	std::cout << "Test ["<<name_test<<"] with " << num_testcases << " testcases: ";
	if (is_passed)
	{
		std::cout << "[PASS]" << std::endl;
	}
	else
	{
		std::cout << "[FAIL]" << std::endl;
	}
	std::cout << "\tAverage error: " << avg_actual << " | Expected average error: " << avg_expected << std::endl;
	std::cout << "\tMaximum error: " << max_actual << " | Expected maximum error: " << max_expected << std::endl;
	std::cout << "\tNumber least-squares solutions: " << num_LS << " expected: 0" << std::endl;
	std::cout << "===== \n time (nanoseconds): " << time_avg << std::endl;

	return is_passed;
}

bool test_SP1(const std::string &csv_path)
{
	std::vector<std::pair<std::string, std::vector<double>>> data;

	// subproblem 1
	data = read_csv(csv_path);
	if (data.size() != 10)
	{
		std::cerr << "Invalid input data for subproblem 1. \n";
		return 0;
	}

	double time_avg{0};
	const unsigned number_testcases{data.at(0).second.size()};
	unsigned number_least_squares{0};

	double max_expected_error{0};
	double avg_expected_error{0};

	double max_error{0};
	double avg_error{0};
	for (unsigned i = 0; i < number_testcases; i++)
	{
		Eigen::Vector3d p1, k, p2;
		double theta{data.at(9).second.at(i)}; // Actual theta
		p1 << data.at(0).second.at(i), data.at(1).second.at(i), data.at(2).second.at(i);
		p2 << data.at(6).second.at(i), data.at(7).second.at(i), data.at(8).second.at(i);
		k << data.at(3).second.at(i), data.at(4).second.at(i), data.at(5).second.at(i);

		IKS::SP1 test_instance(p1, p2, k);
		const auto start = std::chrono::steady_clock::now();
		test_instance.solve();
		const auto end = std::chrono::steady_clock::now();
		time_avg += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

		const double calculated_error{test_instance.error()};
		const double expected_error{test_instance.error(theta)};

		// Update maximum error
		max_expected_error = max_expected_error > expected_error ? max_expected_error : expected_error;
		max_error = max_error > calculated_error ? max_error : calculated_error;
		avg_error += calculated_error;
		avg_expected_error += expected_error;

		if (test_instance.solution_is_ls())
		{
			number_least_squares++;
		}
	}

	avg_error /= number_testcases;
	avg_expected_error /= number_testcases;
	time_avg /= number_testcases;

	return evaluate_test("Subproblem 1", max_expected_error, avg_expected_error, max_error, avg_error, number_least_squares, number_testcases, time_avg);
}

bool test_SP2(const std::string &csv_path)
{
	std::vector<std::pair<std::string, std::vector<double>>> data;

	// subproblem 2
	data = read_csv(csv_path);
	if (data.size() != 14)
	{
		std::cerr << "Invalid input data for subproblem 2. \n";
		return 0;
	}

	double time_avg{0};
	const unsigned number_testcases{data.at(0).second.size()};
	unsigned number_least_squares{0};

	double max_expected_error{0};
	double avg_expected_error{0};

	double max_error{0};
	double avg_error{0};
	for (unsigned i = 0; i < number_testcases; i++)
	{
		Eigen::Vector3d p1, p2, k1, k2;
		double theta_1, theta_2; // Actual theta 1,2
		p1 << data.at(0).second.at(i), data.at(1).second.at(i), data.at(2).second.at(i);
		k1 << data.at(3).second.at(i), data.at(4).second.at(i), data.at(5).second.at(i);
		k2 << data.at(6).second.at(i), data.at(7).second.at(i), data.at(8).second.at(i);
		p2 << data.at(9).second.at(i), data.at(10).second.at(i), data.at(11).second.at(i);
		theta_1 = data[12].second.at(i);
		theta_2 = data[13].second.at(i);

		IKS::SP2 test_instance(p1, p2, k1, k2);
		const auto start = std::chrono::steady_clock::now();
		test_instance.solve();
		const auto end = std::chrono::steady_clock::now();
		time_avg += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

		const double calculated_error{test_instance.error()};
		const double expected_error{test_instance.error(theta_1, theta_2)};

		// Update maximum error
		max_expected_error = max_expected_error > expected_error ? max_expected_error : expected_error;
		max_error = max_error > calculated_error ? max_error : calculated_error;
		avg_error += calculated_error;
		avg_expected_error += expected_error;

		if (test_instance.solution_is_ls())
		{
			number_least_squares++;
		}
	}

	avg_error /= number_testcases;
	avg_expected_error /= number_testcases;
	time_avg /= number_testcases;

	return evaluate_test("Subproblem 2", max_expected_error, avg_expected_error, max_error, avg_error, number_least_squares, number_testcases, time_avg);
}

bool test_SP3(const std::string &csv_path)
{
	std::vector<std::pair<std::string, std::vector<double>>> data;

	// subproblem 3
	data = read_csv(csv_path);
	if (data.size() != 11)
	{
		std::cerr << "Invalid input data for subproblem 3. \n";
		return 0;
	}

	double time_avg{0};
	const unsigned number_testcases{data.at(0).second.size()};
	unsigned number_least_squares{0};

	double max_expected_error{0};
	double avg_expected_error{0};

	double max_error{0};
	double avg_error{0};
	for (unsigned i = 0; i < number_testcases; i++)
	{
		Eigen::Vector3d p1, p2, k;
		double d;
		double theta;
		p1 << data.at(0).second.at(i), data.at(1).second[i], data.at(2).second.at(i);
		p2 << data.at(3).second.at(i), data.at(4).second[i], data.at(5).second.at(i);
		k << data.at(6).second.at(i), data.at(7).second[i], data.at(8).second.at(i);
		d = data[9].second[i];
		theta = data[10].second[i];

		IKS::SP3 test_instance(p1, p2, k, d);
		const auto start = std::chrono::steady_clock::now();
		test_instance.solve();
		const auto end = std::chrono::steady_clock::now();
		time_avg += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

		const double calculated_error{test_instance.error()};
		const double expected_error{test_instance.error(theta)};

		// Update maximum error
		max_expected_error = max_expected_error > expected_error ? max_expected_error : expected_error;
		max_error = max_error > calculated_error ? max_error : calculated_error;
		avg_error += calculated_error;
		avg_expected_error += expected_error;

		if (test_instance.solution_is_ls())
		{
			number_least_squares++;
		}
	}

	avg_error /= number_testcases;
	avg_expected_error /= number_testcases;
	time_avg /= number_testcases;
	return evaluate_test("Subproblem 3", max_expected_error, avg_expected_error, max_error, avg_error, number_least_squares, number_testcases, time_avg);
}

bool test_SP4(const std::string &csv_path)
{
	std::vector<std::pair<std::string, std::vector<double>>> data;

	// subproblem 4
	data = read_csv(csv_path);
	if (data.size() != 11)
	{
		std::cerr << "Invalid input data for subproblem 4. \n";
		return 0;
	}

	double time_avg{0};
	const unsigned number_testcases{data.at(0).second.size()};
	unsigned number_least_squares{0};

	double max_expected_error{0};
	double avg_expected_error{0};

	double max_error{0};
	double avg_error{0};
	for (unsigned i = 0; i < number_testcases; i++)
	{
		Eigen::Vector3d p, k, h;
		double d;
		double theta;
		p << data.at(0).second.at(i), data.at(1).second[i], data.at(2).second.at(i);
		k << data.at(3).second.at(i), data.at(4).second[i], data.at(5).second.at(i);
		h << data.at(6).second.at(i), data.at(7).second[i], data.at(8).second.at(i);
		d = data[9].second[i];
		theta = data[10].second[i];

		IKS::SP4 test_instance(h, p, k, d);
		const auto start = std::chrono::steady_clock::now();
		test_instance.solve();
		const auto end = std::chrono::steady_clock::now();
		time_avg += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

		const double calculated_error{test_instance.error()};
		const double expected_error{test_instance.error(theta)};

		// Update maximum error
		max_expected_error = max_expected_error > expected_error ? max_expected_error : expected_error;
		max_error = max_error > calculated_error ? max_error : calculated_error;
		avg_error += calculated_error;
		avg_expected_error += expected_error;

		if (test_instance.solution_is_ls())
		{
			number_least_squares++;
		}
	}

	avg_error /= number_testcases;
	avg_expected_error /= number_testcases;
	time_avg /= number_testcases;
	return evaluate_test("Subproblem 4", max_expected_error, avg_expected_error, max_error, avg_error, number_least_squares, number_testcases, time_avg);
}

bool test_SP5(const std::string &csv_path)
{
	std::vector<std::pair<std::string, std::vector<double>>> data;

	// subproblem 5
	data = read_csv(csv_path);
	if (data.size() != 24)
	{
		std::cerr << "Invalid input data for subproblem 5. \n";
		return 0;
	}

	double time_avg{0};
	const unsigned number_testcases{data.at(0).second.size()};
	unsigned number_least_squares{0};

	double max_expected_error{0};
	double avg_expected_error{0};

	double max_error{0};
	double avg_error{0};
	for (unsigned i = 0; i < number_testcases; i++)
	{
		Eigen::Vector3d p0, p1, p2, p3, k1, k2, k3;
		double theta_1, theta_2, theta_3;
		p1 << data.at(0).second.at(i), data.at(1).second.at(i), data.at(2).second.at(i);
		p2 << data.at(3).second.at(i), data.at(4).second.at(i), data.at(5).second.at(i);
		p3 << data.at(6).second.at(i), data.at(7).second.at(i), data.at(8).second.at(i);
		k1 << data.at(9).second.at(i), data.at(10).second.at(i), data.at(11).second.at(i);
		k2 << data.at(12).second.at(i), data.at(13).second.at(i), data.at(14).second.at(i);
		k3 << data.at(15).second.at(i), data.at(16).second.at(i), data.at(17).second.at(i);
		p0 << data.at(18).second.at(i), data.at(19).second.at(i), data.at(20).second.at(i);
		theta_1 = data.at(21).second.at(i);
		theta_2 = data.at(22).second.at(i);
		theta_3 = data.at(23).second.at(i);

		IKS::SP5 test_instance(p0, p1, p2, p3, k1, k2, k3);
		const auto start = std::chrono::steady_clock::now();
		test_instance.solve();
		const auto end = std::chrono::steady_clock::now();
		time_avg += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

		const double calculated_error{test_instance.error()};
		const double expected_error{test_instance.error(theta_1, theta_2, theta_3)};

		// Update maximum error
		max_expected_error = max_expected_error > expected_error ? max_expected_error : expected_error;
		max_error = max_error > calculated_error ? max_error : calculated_error;
		avg_error += calculated_error;
		avg_expected_error += expected_error;

		if (test_instance.solution_is_ls())
		{
			number_least_squares++;
		}
	}

	avg_error /= number_testcases;
	avg_expected_error /= number_testcases;
	time_avg /= number_testcases;
	return evaluate_test("Subproblem 5", max_expected_error, avg_expected_error, max_error, avg_error, number_least_squares, number_testcases, time_avg);
}

bool test_SP6(const std::string &csv_path)
{
	std::vector<std::pair<std::string, std::vector<double>>> data;

	// subproblem 6
	data = read_csv(csv_path);
	if (data.size() != 40)
	{
		std::cerr << "Invalid input data for subproblem 6. \n";
		return 0;
	}

	double time_avg{0};
	const unsigned number_testcases{data.at(0).second.size()};
	unsigned number_least_squares{0};

	double max_expected_error{0};
	double avg_expected_error{0};

	double max_error{0};
	double avg_error{0};
	for (unsigned i = 0; i < number_testcases; i++)
	{
		Eigen::Vector3d h1, h2, h3, h4, k1, k2, k3, k4, p1, p2, p3, p4;
		double d1, d2;
		double theta_1, theta_2;
		h1 << data.at(0).second.at(i), data.at(1).second.at(i), data.at(2).second.at(i);
		h2 << data.at(3).second.at(i), data.at(4).second.at(i), data.at(5).second.at(i);
		h3 << data.at(6).second.at(i), data.at(7).second.at(i), data.at(8).second.at(i);
		h4 << data.at(9).second.at(i), data.at(10).second.at(i), data.at(11).second.at(i);

		k1 << data.at(12).second.at(i), data.at(13).second.at(i), data.at(14).second.at(i);
		k2 << data.at(15).second.at(i), data.at(16).second.at(i), data.at(17).second.at(i);
		k3 << data.at(18).second.at(i), data.at(19).second.at(i), data.at(20).second.at(i);
		k4 << data.at(21).second.at(i), data.at(22).second.at(i), data.at(23).second.at(i);

		p1 << data.at(24).second.at(i), data.at(25).second.at(i), data.at(26).second.at(i);
		p2 << data.at(27).second.at(i), data.at(28).second.at(i), data.at(29).second.at(i);
		p3 << data.at(30).second.at(i), data.at(31).second.at(i), data.at(32).second.at(i);
		p4 << data.at(33).second.at(i), data.at(34).second.at(i), data.at(35).second.at(i);

		d1 =  data.at(36).second.at(i);
		d2 =  data.at(37).second.at(i);
		theta_1 = data.at(38).second.at(i);
		theta_2 = data.at(39).second.at(i);

		IKS::SP6 test_instance(h1, h2, h3, h4, k1, k2, k3, k4, p1, p2, p3, p4, d1, d2);
		const auto start = std::chrono::steady_clock::now();
		test_instance.solve();
		const auto end = std::chrono::steady_clock::now();
		time_avg += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

		const double calculated_error{test_instance.error()};
		const double expected_error{test_instance.error(theta_1, theta_2)};

		// Update maximum error
		max_expected_error = max_expected_error > expected_error ? max_expected_error : expected_error;
		max_error = max_error > calculated_error ? max_error : calculated_error;
		avg_error += calculated_error;
		avg_expected_error += expected_error;

		if (test_instance.solution_is_ls())
		{
			number_least_squares++;
		}
	}

	avg_error /= number_testcases;
	avg_expected_error /= number_testcases;
	time_avg /= number_testcases;
	return evaluate_test("Subproblem 6", max_expected_error, avg_expected_error, max_error, avg_error, number_least_squares, number_testcases, time_avg);
}