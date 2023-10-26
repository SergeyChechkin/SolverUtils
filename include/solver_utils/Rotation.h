/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include "Rotation_min.h"
#include "Rotation_full.h"

namespace solver::rotation {

// so3 rotation 
template<typename T>
class Rotation { 
public:
    // point rotation
    static inline Eigen::Vector3<T> f(const Eigen::Vector3<T>& angle_axis, const Eigen::Vector3<T>& pnt) {  
        using std::hypot;
        using std::sin;
        using std::cos;
        using std::fpclassify;
        static const T one = T(1.0);
        
        AngleAxis<T> aa;
        aa.theta_ = hypot(angle_axis[0], angle_axis[1], angle_axis[2]);

        if (FP_ZERO == fpclassify(aa.theta_)) {
            return Rotation_min<T>::f(angle_axis, pnt);  
        } else {
            aa.cos_theta_ = cos(aa.theta_);
            aa.sin_theta_ = sin(aa.theta_);        
            aa.axis_ = angle_axis / aa.theta_;

            return Rotation_full<T>::f(aa, pnt);
        }
    }

    // partial derivative by rotation
    static inline Eigen::Matrix<T, 3, 3> df_daa(const Eigen::Vector3<T>& angle_axis, const Eigen::Vector3<T>& pnt) {    
        using std::hypot;
        using std::sin;
        using std::cos;
        using std::fpclassify;
        static const T one = T(1.0);
        
        AngleAxis<T> aa;
        aa.theta_ = hypot(angle_axis[0], angle_axis[1], angle_axis[2]);

        if (FP_ZERO == fpclassify(aa.theta_)) {
            return Rotation_min<T>::df_daa(pnt);  
        } else {
            aa.cos_theta_ = cos(aa.theta_);
            aa.sin_theta_ = sin(aa.theta_);        
            aa.axis_ = angle_axis / aa.theta_;

            return Rotation_full<T>::df_daa(aa, Rotation_full<T>::f(aa, pnt));
        }
    }

    // partial derivative by point
    static inline Eigen::Matrix<T, 3, 3> df_dpt(const Eigen::Vector3<T>& angle_axis, const Eigen::Vector3<T>& pnt) {    
        using std::hypot;
        using std::sin;
        using std::cos;
        using std::fpclassify;
        static const T one = T(1.0);
        
        AngleAxis<T> aa;
        aa.theta_ = hypot(angle_axis[0], angle_axis[1], angle_axis[2]);

        if (FP_ZERO == fpclassify(aa.theta_)) {
            return Rotation_min<T>::df_dpt(angle_axis);  
        } else {
            aa.cos_theta_ = cos(aa.theta_);
            aa.sin_theta_ = sin(aa.theta_);        
            aa.axis_ = angle_axis / aa.theta_;

            return Rotation_full<T>::df_dpt(aa);
        }
    }
public:
    Rotation(const Eigen::Vector3<T>& angle_axis) {
        using std::hypot;
        using std::sin;
        using std::cos;
        using std::fpclassify;

        ceres::AngleAxisToRotationMatrix<T>(angle_axis.data(), rot_mat_.data());

        AngleAxis<T> aa;
        aa.theta_ = hypot(angle_axis[0], angle_axis[1], angle_axis[2]);

        if (FP_ZERO == fpclassify(aa.theta_)) {
            min_version_ = true; 
            df_dpt_ = Rotation_min<T>::df_dpt(angle_axis); 
        } else {
            min_version_ = false;
            aa.cos_theta_ = cos(aa.theta_);
            aa.sin_theta_ = sin(aa.theta_);        
            aa.axis_ = angle_axis / aa.theta_;
            df_daa_J_ = Rotation_full<T>::df_daa_J(aa);
            df_dpt_ = Rotation_full<T>::df_dpt(aa);
        }
    }

    inline Eigen::Vector3<T> f(const Eigen::Vector3<T>& pnt) const { 
        return rot_mat_ * pnt;
    }

    inline Eigen::Matrix<T, 3, 3> df_daa(const Eigen::Vector3<T>& pnt) const {
        if (min_version_) {
            return Rotation_min<T>::df_daa(pnt);
        } else {
            return Rotation_full<T>::df_daa(df_daa_J_, f(pnt));
        }
    }

    inline Eigen::Matrix<T, 3, 3> df_dpt(const Eigen::Vector3<T>& pnt) const {    
        return df_dpt_;
    }
private:
    Eigen::Matrix3<T> rot_mat_;   
    Eigen::Matrix3<T> df_dpt_;
    bool min_version_ = false;
    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> df_daa_J_;


};

}