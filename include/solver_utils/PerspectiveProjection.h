/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

namespace solver::projection {

template<typename T>
class PerspectiveProjection {
public:
    // unit plane projection
    static Eigen::Vector2<T> f(const Eigen::Vector3<T>& pnt) {
        return {pnt[0] / pnt[2], pnt[1] / pnt[2]};
    }

    static Eigen::Matrix<T, 2, 3> df_dpt(const Eigen::Vector3<T>& pnt) {
        static const T zero = T(0.0);
        const T inv_z = T(1.0) / pnt[2];
        const T inv_z_sqr = -inv_z * inv_z;
        
        Eigen::Matrix<T, 2, 3> result;

        result << inv_z, zero, pnt[0] * inv_z_sqr,
                  zero, inv_z, pnt[1] * inv_z_sqr;

        return result;
    }

    // image projection
    // cam[0] - focal distance, cam[1] - cx, cam[2] - cy  
    static Eigen::Vector2<T> f(const Eigen::Vector3<T>& cam, const Eigen::Vector3<T>& pnt) {
        return {cam[0] * pnt[0] / pnt[2] + cam[1], cam[0] * pnt[1] / pnt[2] + cam[2]};
    }

    static Eigen::Matrix<T, 2, 3> df_dpt(const Eigen::Vector3<T>& cam, const Eigen::Vector3<T>& pnt) {
        static const T zero = T(0.0);
        const T inv_z = T(1.0) / pnt[2];
        const T f_by_z = cam[0] * inv_z;
        const T f_by_z_sqr = -f_by_z * inv_z; 

        Eigen::Matrix<T, 2, 3> result;
        
        result << f_by_z, zero, pnt[0] * f_by_z_sqr, 
                zero, f_by_z, pnt[1] * f_by_z_sqr;
        
        return result;
    }

    static Eigen::Matrix<T, 2, 3> df_dcm(const Eigen::Vector3<T>& cam, const Eigen::Vector3<T>& pnt) {
        static const T zero = T(0.0);
        static const T one = T(1.0);
        const T inv_z = T(1.0) / pnt[2];
        
        Eigen::Matrix<T, 2, 3> result;
        
        result << pnt[0] * inv_z, one, zero, 
                  pnt[1] * inv_z, zero, one;
        
        return result;
    }
};

template<typename T>
class PerspectiveReprojectionPlane {
public:

};



}