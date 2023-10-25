/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include <ceres/rotation.h>
#include <Eigen/Geometry>

namespace solver {

template <typename T>
inline void CrossProduct(const T x[3], const T y[3], T x_cross_y[3]) {
  x_cross_y[0] = x[1] * y[2] - x[2] * y[1];
  x_cross_y[1] = x[2] * y[0] - x[0] * y[2];
  x_cross_y[2] = x[0] * y[1] - x[1] * y[0];
}

template <typename T>
inline Eigen::Transform<T, 3, Eigen::Isometry> ExpSE3(const Eigen::Vector<T, 6> pose) {
    Eigen::Matrix<T, 3, 3> rot_mat;
    ceres::AngleAxisToRotationMatrix<T>(pose.data(), rot_mat.data());

    Eigen::Transform<T, 3, Eigen::Isometry> result;
    result.linear() = rot_mat;
    result.translation() << pose[3], pose[4], pose[5];

    return result;
}

template <typename T>
inline Eigen::Vector<T, 6> LogSE3(const Eigen::Transform<T, 3, Eigen::Isometry>& ism_pose) {
    Eigen::Vector<T, 6> result;
    const Eigen::Matrix<T, 3, 3> rot_mat = ism_pose.linear();
    const Eigen::Vector3<T> t = ism_pose.translation();

    ceres::RotationMatrixToAngleAxis<T>(rot_mat.data(), result.data());
    result[3] = t[0]; 
    result[4] = t[1]; 
    result[5] = t[2]; 

    return result;
}

}