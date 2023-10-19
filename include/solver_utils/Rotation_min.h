#pragma once

#include <Eigen/Core>

namespace solver::rotation {

// so3 rotation for very small roatation angle 
template<typename T>
class Rotation_min { 
public:
    // point rotation
    static Eigen::Vector3<T> f(const Eigen::Vector3<T>& angle_axis, const Eigen::Vector3<T>& pnt) {   
        Eigen::Vector3<T> result;
        // (I + aa^) * pt
        result << pnt[0] + angle_axis[1] * pnt[2] - angle_axis[2] * pnt[1],
                  pnt[1] + angle_axis[2] * pnt[0] - angle_axis[0] * pnt[2],
                  pnt[2] + angle_axis[0] * pnt[1] - angle_axis[1] * pnt[0];
        return result;
    }

    // partial derivative by rotation
    static Eigen::Matrix<T, 3, 3> df_daa(const Eigen::Vector3<T>& pnt) {    
        static const T zero = T(0.0);
        Eigen::Matrix<T, 3, 3> result;
        // -pt^ 
        result << zero, pnt[2], -pnt[1], 
                  -pnt[2], zero, pnt[0], 
                  pnt[1], -pnt[0], zero;
        return result;
    }

    // partial derivative by point
    static Eigen::Matrix<T, 3, 3> df_dpt(const Eigen::Vector3<T>& angle_axis) {    
        static const T one = T(1.0);
        Eigen::Matrix<T, 3, 3> result;
        // I + aa^
        result << one, -angle_axis[2], angle_axis[1], 
                  angle_axis[2], one, -angle_axis[0], 
                  -angle_axis[1], angle_axis[0], one;
        return result;
    }
};

}