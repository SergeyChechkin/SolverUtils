// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include <Eigen/Core>
namespace solver {

class BASolver {
public:
    static void SolvePosePointsCeres(
        const std::vector<Eigen::Vector2d>& features_0,
        const std::vector<Eigen::Vector2d>& features_1, 
        Eigen::Vector<double, 6>& pose, 
        std::vector<Eigen::Vector3d>& points);

    static void SolvePosePoints(
        const std::vector<Eigen::Vector2d>& features_0,
        const std::vector<Eigen::Vector2d>& features_1, 
        Eigen::Vector<double, 6>& pose, 
        std::vector<Eigen::Vector3d>& points);
};

}