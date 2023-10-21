/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include <Eigen/Core>
namespace solver {

// Perspective-n-Point transformation solver (Gauss–Newton algorithm).
// Required less than 30 degree rotation error initial guess from optimal solution in order to converge.   
class PnPSolver {
public:
    struct Cofiguration {
        double min_cost = 1.0E-6;
        double min_cost_change = 1.0E-8;
        size_t max_iterations = 150;
        bool verbal = true;
    };

    struct Report {
        double cost;
        size_t iterations;
    };

    static Report SolvePose(
        const std::vector<Eigen::Vector3d>& points,
        const std::vector<Eigen::Vector2d>& features, 
        Eigen::Vector<double, 6>& pose, 
        const Cofiguration& config);
private:
    // Perspective-n-Point factor, direct error between 3D point projection and feature 
    // point - 3D point
    // feature - unit plane projection 
    // pose - frame pose
    // H - Hessian, Jt x J   
    // b - -et x J 
    // cost - et x e; 
    static inline void GetPoseFactor(
        const Eigen::Vector3d& point,
        const Eigen::Vector2d& feature, 
        const Eigen::Vector<double, 6>& pose,
        Eigen::Matrix<double, 2, 6>& J,
        Eigen::Matrix<double, 2, 1>& error);
};

}

