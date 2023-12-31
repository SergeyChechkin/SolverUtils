/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once
#include "solver_utils/Transformation.h"
#include <Eigen/Core>

namespace solver {

// Perspective-n-Point transformation solver with epipolar constraints (Gauss–Newton algorithm).
// Depth of 3D points is undefined 
class EpipolarPnPSolver {
public:
    struct Cofiguration {
        double min_cost = 1.0E-7;
        double min_cost_change = 1.0E-8;
        size_t max_iterations = 150;
        bool verbal = true;
    };

    struct Report {
        double min_cost;
        size_t iterations;
    };

    static Report SolvePose(
        const std::vector<Eigen::Vector3d>& points,
        const std::vector<Eigen::Vector2d>& points_info,
        const std::vector<Eigen::Vector2d>& features, 
        Eigen::Vector<double, 6>& pose, 
        const Cofiguration& config);
    
    static Report SolvePosePoints(
        const std::vector<Eigen::Vector2d>& features_0, // static frame 
        const std::vector<Eigen::Vector2d>& features_1, // transformed frame
        std::vector<Eigen::Vector3d>& points,
        Eigen::Vector<double, 6>& pose, 
        const Cofiguration& config);

private:
    static inline void GetPoseFactor(
        const Eigen::Vector3d& point,
        const Eigen::Vector2d& point_info,
        const Eigen::Vector2d& feature, 
        const transformation::IsometricTransformation<double>& pose,
        Eigen::Matrix<double, 2, 6>& J,
        Eigen::Matrix<double, 2, 1>& error);
};

}