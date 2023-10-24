// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include <Eigen/Core>
namespace solver {

class HomographySolver {
public:
    static bool SolvePoseCeres(
        const std::vector<Eigen::Vector2d>& features_0,
        const std::vector<Eigen::Vector2d>& features_1, 
        Eigen::Vector<double, 6>& pose);
};

}