/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include "Rotation.h"

namespace solver::transformation {

template<typename T>
class Transformation {
public:
    static Eigen::Vector3<T> f(const Eigen::Vector<T, 6>& pose, const Eigen::Vector3<T>& pnt) { 
        Eigen::Vector3<T> result = rotation::Rotation<T>::f(pose.head(3), pnt);
        result[0] += pose[3];
        result[1] += pose[4];
        result[2] += pose[5];

        return result;
    }

    // partial derivative by pose
    static Eigen::Matrix<T, 3, 6> df_dps(const Eigen::Vector<T, 6>& pose, const Eigen::Vector3<T>& pnt) {  
        Eigen::Matrix<T, 3, 6> result;

        result.block(0, 0, 3, 3) = rotation::Rotation<T>::df_daa(pose.head(3), pnt);
        result.block(0, 3, 3, 3) = Eigen::Matrix3<T>::Identity();

        return result; 
    }

    // partial derivative by point
    static Eigen::Matrix<T, 3, 3> df_dpt(const Eigen::Vector<T, 6>& pose, const Eigen::Vector3<T>& pnt) {   
        return rotation::Rotation<T>::df_dpt(pose.head(3), pnt);
    }

    // partial derivative by zero pose
    inline static Eigen::Matrix<T, 3, 6> df_dps_zero(const Eigen::Vector3<T>& pnt) {
        static const T one = T(1.0);
        Eigen::Matrix<T, 3, 6> result;
        
        result.setZero();

        result(0,1) = pnt[2];
        result(0,2) = -pnt[1];
        result(0,3) = one;

        result(1,0) = -pnt[2];
        result(1,2) = pnt[0];
        result(1,4) = one;

        result(2,0) = pnt[1];
        result(2,1) = -pnt[0];
        result(2,5) = one;

        return result;
    }
}; 

}