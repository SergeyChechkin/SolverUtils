/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include "Rotation.h"
#include "Functions.h"

#include <Eigen/Geometry>

namespace solver::transformation {

template<typename T>
class IsometricTransformation {
public:
    static inline Eigen::Vector3<T> f(const Eigen::Vector<T, 6>& pose, const Eigen::Vector3<T>& pnt) { 
        Eigen::Vector3<T> result = rotation::Rotation<T>::f(pose.head(3), pnt);
        result[0] += pose[3];
        result[1] += pose[4];
        result[2] += pose[5];

        return result;
    }

    static inline Eigen::Vector3<T> f(const Eigen::Transform<T, 3, Eigen::Isometry>& ism_pose, const Eigen::Vector3<T>& pnt) { 
        return ism_pose * pnt;
    }

    // partial derivative by pose
    static inline Eigen::Matrix<T, 3, 6> df_dps(const Eigen::Vector<T, 6>& pose, const Eigen::Vector3<T>& pnt) {  
        Eigen::Matrix<T, 3, 6> result;

        result.block(0, 0, 3, 3) = rotation::Rotation<T>::df_daa(pose.head(3), pnt);
        result.block(0, 3, 3, 3) = Eigen::Matrix3<T>::Identity();

        return result; 
    }

    // partial derivative by point
    static inline Eigen::Matrix<T, 3, 3> df_dpt(const Eigen::Vector<T, 6>& pose, const Eigen::Vector3<T>& pnt) {   
        return rotation::Rotation<T>::df_dpt(pose.head(3), pnt);
    }

    // partial derivative by zero pose
    static inline Eigen::Matrix<T, 3, 6> df_dps_zero(const Eigen::Vector3<T>& pnt) {
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
public:
    IsometricTransformation(const Eigen::Vector<T, 6>& pose)
    : rot_(pose.head(3)) 
    {
        pose_ = ExpSE3(pose);      
    }

    inline Eigen::Vector3<T> f(const Eigen::Vector3<T>& pnt) const  { 
        return pose_ * pnt;
    }

    inline Eigen::Matrix<T, 3, 6> df_dps(const Eigen::Vector3<T>& pnt) const {
        Eigen::Matrix<T, 3, 6> result;

        result.block(0, 0, 3, 3) = rot_.df_daa(pnt);
        result.block(0, 3, 3, 3) = Eigen::Matrix3<T>::Identity();

        return result; 
    }

    inline Eigen::Matrix<T, 3, 3> df_dpt(const Eigen::Vector3<T>& pnt) const {
        return rot_.df_dpt(pnt); 
    }

    inline Eigen::Vector3<T> translation() const  { 
        return pose_.translation();
    }

    inline Eigen::Matrix3<T> rotation() const  { 
        return pose_.linear();
    }
    
private:
    rotation::Rotation<T> rot_;
    Eigen::Transform<T, 3, Eigen::Isometry> pose_;
}; 

}