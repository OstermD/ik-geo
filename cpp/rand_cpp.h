#ifndef __rand_cpp__
#define __rand_cpp__

#include <time.h>
#include <math.h>
#include <random>
#include <eigen3/Eigen/Dense>
#include <vector>

namespace RAND_CPP
{
  //Create number from 0.0000 to 1.0000
  double rand_0to1();

  //Example input is a normal vector
  //Matrix cross-product for 3 x 3 vector
  Eigen::Matrix<double, 3, 3> hat(const Eigen::Matrix<double, 3, Eigen::Dynamic>& vec);

  //Create a random angle
  double rand_angle();

  Eigen::Matrix<double, Eigen::Dynamic, 1> rand_angle(int N);

  //Create random matrix of width = size, height = 3
  Eigen::Matrix<double, 3, Eigen::Dynamic> rand_vec(int N = 1);

  Eigen::Matrix<double, 3, Eigen::Dynamic> rand_normal_vec(int size = 1);

  // Only works with input vectors of size 1.  Same as MATLAB implementation
  Eigen::VectorXd rand_perp_normal_vec(const Eigen::Vector3d& vec);

  //Create 3x3 rotation matrix using the Euler Rodrigues formula
  //TODO: Compare notes
  Eigen::Matrix<double, 3, Eigen::Dynamic> rot(Eigen::Matrix<double, 3, Eigen::Dynamic> k, double theta);
}
#endif