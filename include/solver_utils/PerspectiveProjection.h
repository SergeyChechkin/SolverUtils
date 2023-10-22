/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

namespace solver::projection {

template<typename T>
class PerspectiveProjection {
public:
    // unit plane projection
    static inline Eigen::Vector2<T> f(const Eigen::Vector3<T>& pnt) {
        return {pnt[0] / pnt[2], pnt[1] / pnt[2]};
    }

    // Projection derivative by point
    static inline Eigen::Matrix<T, 2, 3> df_dpt(const Eigen::Vector3<T>& pnt) {
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
    static inline Eigen::Vector2<T> f(const Eigen::Vector3<T>& cam, const Eigen::Vector3<T>& pnt) {
        return {cam[0] * pnt[0] / pnt[2] + cam[1], cam[0] * pnt[1] / pnt[2] + cam[2]};
    }

    // Projection derivative by point
    static inline Eigen::Matrix<T, 2, 3> df_dpt(const Eigen::Vector3<T>& cam, const Eigen::Vector3<T>& pnt) {
        static const T zero = T(0.0);
        const T inv_z = T(1.0) / pnt[2];
        const T f_by_z = cam[0] * inv_z;
        const T f_by_z_sqr = -f_by_z * inv_z; 

        Eigen::Matrix<T, 2, 3> result;
        
        result << f_by_z, zero, pnt[0] * f_by_z_sqr, 
                zero, f_by_z, pnt[1] * f_by_z_sqr;
        
        return result;
    }

    // Projection derivative by camera parameters
    static inline Eigen::Matrix<T, 2, 3> df_dcm(const Eigen::Vector3<T>& cam, const Eigen::Vector3<T>& pnt) {
        static const T zero = T(0.0);
        static const T one = T(1.0);
        const T inv_z = T(1.0) / pnt[2];
        
        Eigen::Matrix<T, 2, 3> result;
        
        result << pnt[0] * inv_z, one, zero, 
                  pnt[1] * inv_z, zero, one;
        
        return result;
    }
};

// Perspective reprojection
template<typename T>
class PerspectiveReprojectionUnitPlane {
public:
// reprojection from unit plane 
static inline Eigen::Vector3<T> f(const Eigen::Vector2<T>& pnt, T inv_depth = T(1.0)) {
        const T depth = T(1.0) / inv_depth;
        return {depth * pnt[0], depth * pnt[1], depth};
    }

// reprojection derivatives by point
static inline Eigen::Matrix<T, 3, 2> df_dpt(const Eigen::Vector2<T>& pnt, T inv_depth = T(1.0)) {
        Eigen::Matrix<T, 3, 2> result = Eigen::Matrix<T, 3, 2>::Zero();
        const T depth = T(1.0) / inv_depth;
        
        result(0, 0) = depth;
        result(1, 1) = depth;

        return result;
    }

// reprojection derivatives by inverse depth
static inline Eigen::Matrix<T, 3, 1> df_did(const Eigen::Vector2<T>& pnt, T inv_depth = T(1.0)) {
        const T did = -T(1.0) / (inv_depth * inv_depth);
        return {did * pnt[0], did * pnt[1], did};
    }

// reprojection from image
static inline Eigen::Vector3<T> f(const Eigen::Vector3<T>& cam, const Eigen::Vector2<T>& pnt, T inv_depth = T(1.0)) {
        const T depth = T(1.0) / inv_depth;
        const T inv_f = T(1.0) / cam[0];
        return {depth * inv_f * (pnt[0] - cam[1]), depth * inv_f * (pnt[1] - cam[2]), depth};
    }

// reprojection derivatives by camera parameters
static inline Eigen::Matrix<T, 3, 3> df_dcm(const Eigen::Vector3<T>& cam, const Eigen::Vector2<T>& pnt, T inv_depth = T(1.0)) {
        static const T zero = T(0.0);

        Eigen::Matrix<T, 3, 3> result = Eigen::Matrix<T, 3, 3>::Zero();
        const T depth = T(1.0) / inv_depth;
        const T inv_f = T(1.0) / cam[0];
        const T inv_df = -T(1.0) / (cam[0] * cam[0]);

        result(0, 0) = depth * inv_df * (pnt[0] - cam[1]); 
        result(0, 1) = -depth * inv_f; 
        result(1, 0) = depth * inv_df * (pnt[1] - cam[2]); 
        result(1, 2) = -depth * inv_f; 

        return result;
    }

// reprojection derivatives by point
static inline Eigen::Matrix<T, 3, 2> df_dpt(const Eigen::Vector3<T>& cam, const Eigen::Vector2<T>& pnt, T inv_depth = T(1.0)) {
        Eigen::Matrix<T, 3, 2> result = Eigen::Matrix<T, 3, 2>::Zero();

        const T depth = T(1.0) / inv_depth;
        const T inv_f = T(1.0) / cam[0];

        result(0, 0) = depth * inv_f; 
        result(1, 1) = depth * inv_f; 

        return result;
    }

// reprojection derivatives by inverse depth
static inline Eigen::Matrix<T, 3, 1> df_did(const Eigen::Vector3<T>& cam, const Eigen::Vector2<T>& pnt, T inv_depth = T(1.0)) {
        const T did = -T(1.0) / (inv_depth * inv_depth);
        const T inv_f = T(1.0) / cam[0];
        return {did * inv_f * (pnt[0] - cam[1]), did * inv_f * (pnt[1] - cam[2]), did};
    }
};



}