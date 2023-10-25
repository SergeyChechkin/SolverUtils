/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include "Functions.h"
#include <Eigen/Core>

namespace solver::rotation {

// Precomputed so3 rotation parameters
template<typename T>
struct AngleAxis {
    AngleAxis() {}
    explicit AngleAxis(const Eigen::Vector3<T>& angle_axis)  {
        using std::hypot;
        using std::sin;
        using std::cos;

        theta_ = hypot(angle_axis[0], angle_axis[1], angle_axis[2]);
        cos_theta_ = cos(theta_);
        sin_theta_ = sin(theta_);        
        axis_ = angle_axis / theta_;
    }
    Eigen::Vector3<T> axis_;
    T theta_;
    T sin_theta_;
    T cos_theta_;
};

// so3 rotation
template<typename T>
class Rotation_full { 
public:
    // point rotation
    static inline Eigen::Vector3<T> f(const AngleAxis<T>& aa, const Eigen::Vector3<T>& pnt) {   
        static const T one = T(1.0);
        Eigen::Vector3<T> result;
        const T a_cross_pt[3] = {aa.axis_[1] * pnt[2] - aa.axis_[2] * pnt[1],
                                aa.axis_[2] * pnt[0] - aa.axis_[0] * pnt[2],
                                aa.axis_[0] * pnt[1] - aa.axis_[1] * pnt[0]};

        // a_t * pt * (1-cos(theta))
        const T one_costh = (one - aa.cos_theta_);
        const T tmp = (aa.axis_[0] * pnt[0] + aa.axis_[1] * pnt[1] + aa.axis_[2] * pnt[2]) * one_costh;

        result << pnt[0] * aa.cos_theta_ + a_cross_pt[0] * aa.sin_theta_ + aa.axis_[0] * tmp,
                  pnt[1] * aa.cos_theta_ + a_cross_pt[1] * aa.sin_theta_ + aa.axis_[1] * tmp,
                  pnt[2] * aa.cos_theta_ + a_cross_pt[2] * aa.sin_theta_ + aa.axis_[2] * tmp;

        return result;
    }

    // partial derivative by rotation
    static inline Eigen::Matrix<T, 3, 3> df_daa(const AngleAxis<T>& aa, const Eigen::Vector3<T>& rot_pnt) {  
        static const T one = T(1.0);
        // -(Rp)^J 
        const T c0 = aa.sin_theta_ / aa.theta_;
        const T c1 = one - c0;
        const T c2 = (one - aa.cos_theta_) / aa.theta_;

        const T w00 = c1 * aa.axis_[0] * aa.axis_[0];
        const T w01 = c1 * aa.axis_[0] * aa.axis_[1];
        const T w02 = c1 * aa.axis_[0] * aa.axis_[2];
        const T w11 = c1 * aa.axis_[1] * aa.axis_[1];
        const T w12 = c1 * aa.axis_[1] * aa.axis_[2];
        const T w22 = c1 * aa.axis_[2] * aa.axis_[2];

        const T c2w0 = c2 * aa.axis_[0];
        const T c2w1 = c2 * aa.axis_[1];
        const T c2w2 = c2 * aa.axis_[2];
/*        
        const T J[9] = {c0 + w00, w01 + c2w2, w02 - c2w1, 
                        w01 - c2w2, c0 + w11, w12 + c2w0,
                        w02 + c2w1, w12 - c2w0, c0 + w22};

        Eigen::Matrix<T, 3, 3> result;

        CrossProduct(J, rot_pnt.data(), result.data());
        CrossProduct(J + 3, rot_pnt.data(), result.data() + 3);
        CrossProduct(J + 6, rot_pnt.data(), result.data() + 6);  
*/

        Eigen::Matrix<T, 3, 3, Eigen::RowMajor> J;
        J << c0 + w00, w01 + c2w2, w02 - c2w1, 
             w01 - c2w2, c0 + w11, w12 + c2w0,
             w02 + c2w1, w12 - c2w0, c0 + w22;

        Eigen::Matrix<T, 3, 3> result;

        CrossProduct(J.row(0).data(), rot_pnt.data(), result.data());
        CrossProduct(J.row(1).data(), rot_pnt.data(), result.data() + 3);
        CrossProduct(J.row(2).data(), rot_pnt.data(), result.data() + 6);  

        return result;
    }

    static inline Eigen::Matrix<T, 3, 3, Eigen::RowMajor> df_daa_J(const AngleAxis<T>& aa) {  
        static const T one = T(1.0);
        // -(Rp)^J 
        const T c0 = aa.sin_theta_ / aa.theta_;
        const T c1 = one - c0;
        const T c2 = (one - aa.cos_theta_) / aa.theta_;

        const T w00 = c1 * aa.axis_[0] * aa.axis_[0];
        const T w01 = c1 * aa.axis_[0] * aa.axis_[1];
        const T w02 = c1 * aa.axis_[0] * aa.axis_[2];
        const T w11 = c1 * aa.axis_[1] * aa.axis_[1];
        const T w12 = c1 * aa.axis_[1] * aa.axis_[2];
        const T w22 = c1 * aa.axis_[2] * aa.axis_[2];

        const T c2w0 = c2 * aa.axis_[0];
        const T c2w1 = c2 * aa.axis_[1];
        const T c2w2 = c2 * aa.axis_[2];

        Eigen::Matrix<T, 3, 3, Eigen::RowMajor> J;
        J << c0 + w00, w01 + c2w2, w02 - c2w1, 
             w01 - c2w2, c0 + w11, w12 + c2w0,
             w02 + c2w1, w12 - c2w0, c0 + w22;

        return J;
    }

    static inline Eigen::Matrix<T, 3, 3> df_daa(
        const Eigen::Matrix<T, 3, 3, Eigen::RowMajor>& df_daa_J, 
        const Eigen::Vector3<T>& rot_pnt) { 
            Eigen::Matrix<T, 3, 3> result;

            CrossProduct(df_daa_J.row(0).data(), rot_pnt.data(), result.data());
            CrossProduct(df_daa_J.row(1).data(), rot_pnt.data(), result.data() + 3);
            CrossProduct(df_daa_J.row(2).data(), rot_pnt.data(), result.data() + 6);  

            return result;
        }

    // partial derivative by point
    static inline Eigen::Matrix<T, 3, 3> df_dpt(const AngleAxis<T>& aa) {  
        static const T one = T(1.0);
        const T one_costh = (one - aa.cos_theta_);
        const T dtmp_dpt[3] = {aa.axis_[0] * one_costh, aa.axis_[1] * one_costh, aa.axis_[2] * one_costh};

        Eigen::Matrix<T, 3, 3> result;

        result(0, 0) = aa.cos_theta_ + aa.axis_[0] * dtmp_dpt[0];
        result(1, 0) = aa.axis_[2] * aa.sin_theta_ + aa.axis_[1] * dtmp_dpt[0];
        result(2, 0) = - aa.axis_[1] * aa.sin_theta_ + aa.axis_[2] * dtmp_dpt[0];

        result(0, 1) = - aa.axis_[2] * aa.sin_theta_ + aa.axis_[0] * dtmp_dpt[1];
        result(1, 1) = aa.cos_theta_ + aa.axis_[1] * dtmp_dpt[1];
        result(2, 1) = aa.axis_[0] * aa.sin_theta_ + aa.axis_[2] * dtmp_dpt[1];

        result(0, 2) = aa.axis_[1] * aa.sin_theta_ + aa.axis_[0] * dtmp_dpt[2];
        result(1, 2) = - aa.axis_[0] * aa.sin_theta_ + aa.axis_[1] * dtmp_dpt[2];
        result(2, 2) = aa.cos_theta_ + aa.axis_[2] * dtmp_dpt[2]; 

        return result; 
    }
};

}