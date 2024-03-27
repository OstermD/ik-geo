#include <eigen3/Eigen/Dense>
#include <vector>
#include <iostream>
#include <math.h>

#include "../subproblems/sp.h"
#include "IKS.h"

namespace IKS
{
    General_Robot::General_Robot(const Eigen::Matrix<double, 3, 6> &H, const Eigen::Matrix<double, 3, 7> &P)
        : H(H), P(P)
    {
    }

    IK_Solution General_Robot::calculate_IK(const Homogeneous_T &ee_position_orientation) const
    {
        IK_Solution solution;

        // New Approach -> Two consecutive intersecting axes: h5 and h6
        const Eigen::Vector3d p_0t = ee_position_orientation.block<3, 1>(0, 3);
        const Eigen::Matrix3d r_06 = ee_position_orientation.block<3, 3>(0, 0);
        const Eigen::Vector3d p_16 = p_0t - this->P.col(0) - r_06 * this->P.col(6);

        // Step 1: Get R_04 via SP$

        const Eigen::Vector3d h6_rotated = r_06*this->H.col(5);
        SP4 sp4(this->H.col(4), h6_rotated, -this->H.col(3), this->H.col(4).transpose()*this->H.col(5));
        sp4.solve();
        const std::vector<double> theta_40 = sp4.get_theta();

        for(const auto& q40 : theta_40)
        {
            const Eigen::Matrix3d r_40 = Eigen::AngleAxisd(q40, this->H.col(0).normalized()).toRotationMatrix();
            SP1 sp1_q5(this->H.col(5), r_40*r_06*this->H.col(5), this->H.col(4));
            sp1_q5.solve();
            const double q5{sp1_q5.get_theta()};


            SP1 sp1_q6(this->H.col(4), r_06.transpose()*r_40.transpose()*this->H.col(4), -this->H.col(5));
            sp1_q6.solve();

            const double q6{sp1_q6.get_theta()};

            const Eigen::Matrix3d r_45 = Eigen::AngleAxisd(q5, this->H.col(4).normalized()).toRotationMatrix();
            const Eigen::Matrix3d r_05 = r_40.transpose()*r_45;
            SP5 sp5(Eigen::Vector3d(0,0,0), 
                    this->H.col(4), 
                    Eigen::Vector3d(0,0,0), 
                    r_05*this->H.col(4),
                    this->H.col(3),
                    this->H.col(2),
                    this->H.col(1));
            sp5.solve();

            const std::vector<double> &theta4 = sp5.get_theta_1();
            const std::vector<double> &theta3 = sp5.get_theta_2();
            const std::vector<double> &q02 = sp5.get_theta_3();

            if (theta4.size() != theta3.size() || theta3.size() != q02.size())
            {
                throw std::runtime_error("Invalid number of angle combinations gathered from SP5!");
            }

            for(unsigned i = 0; i < theta4.size(); ++i)
            {
                const double& q4{theta4.at(i)};
                const double& q3{theta3.at(i)};

                const Eigen::Matrix3d r_23 = Eigen::AngleAxisd(q3, this->H.col(2).normalized()).toRotationMatrix();
                const Eigen::Matrix3d r_34 = Eigen::AngleAxisd(q4, this->H.col(3).normalized()).toRotationMatrix();
                const Eigen::Matrix3d r_56 = Eigen::AngleAxisd(q6, this->H.col(5).normalized()).toRotationMatrix();

                const Eigen::Matrix3d r_26 = r_23*r_34*r_56;

                SP1 sp1_q1(this->H.col(1), -r_06*r_26.transpose()*this->H.col(2), -this->H.col(0));
                sp1_q1.solve();

                const Eigen::Matrix3d r_01 = Eigen::AngleAxisd(q6, this->H.col(5).normalized()).toRotationMatrix();
                SP1 sp1_q2(this->H.col(0), r_26*r_06.transpose()*this->H.col(0), -this->H.col(1));
                sp1_q2.solve();

                solution.Q.push_back({sp1_q1.get_theta(),sp1_q2.get_theta(),q3,q4,q5,q6});
            }
        }


        /*
        const Eigen::Vector3d p_0t = ee_position_orientation.block<3, 1>(0, 3);
        const Eigen::Matrix3d r_06 = ee_position_orientation.block<3, 3>(0, 0);
        const Eigen::Vector3d p_16 = p_0t - this->P.col(0) - r_06 * this->P.col(6);

        // Calculate "Position-IK":
        std::vector<std::vector<double>> position_solutions;

        // Check for parallel axes
        if (this->H.col(0).cross(this->H.col(1)).norm() < ZERO_THRESH &&
            this->H.col(0).cross(this->H.col(2)).norm() < ZERO_THRESH &&
            this->H.col(1).cross(this->H.col(2)).norm() < ZERO_THRESH)
        {
            // h1 || h2 || h3 -> first three axes parallel
        }
        else if (this->H.col(1).cross(this->H.col(2)).norm() < ZERO_THRESH &&
                 this->H.col(1).cross(this->H.col(3)).norm() < ZERO_THRESH &&
                 this->H.col(2).cross(this->H.col(3)).norm() < ZERO_THRESH)
        {
            // h2 || h3 || h4
            const double d1 = this->H.col(1).transpose() * (this->P.col(2) + this->P.col(3) + this->P.col(4) + this->P.col(1));
            const double d2 = 0;

            SP6 sp6(this->H.col(1),
                    this->H.col(1),
                    this->H.col(1),
                    this->H.col(1),
                    -this->H.col(0),
                    this->H.col(4),
                    -this->H.col(0),
                    this->H.col(4),
                    p_16,
                    -this->P.col(5),
                    r_06 * this->H.col(5),
                    -this->H.col(5),
                    d1,
                    d2);

            sp6.solve();
            const std::vector<double> theta_1 = sp6.get_theta_1();
            const std::vector<double> theta_5 = sp6.get_theta_2();

            for(unsigned i = 0; i < theta_1.size(); i++)
            {
                const double& q1 = theta_1.at(i);
                const double& q5 = theta_5.at(i);

                const Eigen::Matrix3d r_01 = Eigen::AngleAxisd(q1, this->H.col(0).normalized()).toRotationMatrix();
                const Eigen::Matrix3d r_45 = Eigen::AngleAxisd(q5, this->H.col(4).normalized()).toRotationMatrix();

                SP1 sp_14(r_45*this->H.col(5), r_01.transpose()*r_06*this->H.col(5), this->H.col(1));
                SP1 sp_q6(r_45.transpose()*this->H.col(1), r_06.transpose()*r_01*this->H.col(1), -this->H.col(5));
                sp_14.solve();

                const Eigen::Matrix3d r_14 = Eigen::AngleAxisd(sp_14.get_theta(), this->H.col(1).normalized()).toRotationMatrix();
                const Eigen::Vector3d d_inner = r_01.transpose()*p_16-this->P.col(1) - r_14*r_45*this->P.col(5) - r_14*this->P.col(4);
                const double d = d_inner.norm();

                SP3 sp3_t3(-this->P.col(3), this->P.col(2), this->H.col(1), d);
                sp3_t3.solve();
                sp_q6.solve();

                const std::vector<double> theta_3 = sp3_t3.get_theta();

                for(const auto& q3 : theta_3)
                {                
                    const Eigen::Matrix3d rot_1 = Eigen::AngleAxisd(q3, this->H.col(1).normalized()).toRotationMatrix();
                    SP1 sp1_q2(this->P.col(2)+rot_1*this->P.col(3), d_inner, this->H.col(1));
                    sp1_q2.solve();

                    double q4 = sp_14.get_theta() - sp1_q2.get_theta() - q3;
                    q4 = std::atan2(std::sin(q4), std::cos(q4)); // Map angle q4 to [-PI,PI)

                    solution.Q.push_back({q1, sp1_q2.get_theta(), q3, q4, q5, sp_q6.get_theta()});
                    solution.is_LS_vec.push_back(sp6.solution_is_ls() || sp1_q2.solution_is_ls() || sp3_t3.solution_is_ls() || sp_q6.solution_is_ls());
                }
            }
        }
        else if (this->H.col(2).cross(this->H.col(3)).norm() < ZERO_THRESH &&
                 this->H.col(2).cross(this->H.col(4)).norm() < ZERO_THRESH &&
                 this->H.col(3).cross(this->H.col(4)).norm() < ZERO_THRESH)
        {
            // h3 || h4 || h5
        }
        else if (this->H.col(3).cross(this->H.col(4)).norm() < ZERO_THRESH &&
                 this->H.col(3).cross(this->H.col(5)).norm() < ZERO_THRESH &&
                 this->H.col(4).cross(this->H.col(5)).norm() < ZERO_THRESH)
        {
            // h4 || h5 || h6
        }

        // Solve "orientation IK"
        */
        return solution;
    }

    Homogeneous_T General_Robot::fwdkin(const std::vector<double> &Q) const
    {
        return fwd_kinematics_ndof(this->H, this->P, Q);
    }
};