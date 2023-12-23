#pragma once

#include <Eigen/Core>

namespace unscented_update {
    
template<typename T, size_t src_Dm, size_t dst_Dm>
class UpdateFunction {
public:
    static constexpr size_t src_dim = src_Dm;  // Input distrebution dimension 
    static constexpr size_t dst_dim = dst_Dm;  // Output distrebution dimension 
    using ScalarT = T;
    using SrcPointT = Eigen::Vector<T, src_dim>;
    using DstPointT = Eigen::Vector<T, dst_dim>;
public:
    UpdateFunction() : scale_(1.0) {
    }

    UpdateFunction(T scale) : scale_(scale) {
    }

    DstPointT operator()(const SrcPointT& src) const {
        return scale_ * src;
    } 
private:
    T scale_ = 1;
};

}