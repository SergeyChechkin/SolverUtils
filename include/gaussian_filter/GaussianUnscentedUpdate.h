#pragma once

#include "gaussian_filter/GaussianDistribution.h"

namespace gaussian_filter {

template<typename FuncT>
class GaussianUnscentedUpdate {
public:
    static constexpr size_t src_dim = FuncT::src_dim;
    static constexpr size_t dst_dim = FuncT::dst_dim; 
    using ScalarT = FuncT::ScalarT;
    using SrcDistT = GaussianDistribution<ScalarT, src_dim>; // Input gaussian distrebution
    using DstDistT = GaussianDistribution<ScalarT, dst_dim>; // Output gaussian distrebution
private:
    static constexpr size_t pnts_size = 2 * src_dim; // Number of generated points, exclude mean
    using SrcPointT = FuncT::SrcPointT;
    using DstPointT = FuncT::DstPointT;
    using SrcCovarSqerT = typename SrcDistT::MatT;
    using DstMeanT = typename SrcDistT::MeanT;
    using DstCovarT = typename SrcDistT::MatT;

    // Unscented update parameters
    //static constexpr ScalarT alpha = 1;
    //static constexpr ScalarT betta = 0;
    //static constexpr ScalarT kappa = 0;

    static constexpr ScalarT alpha = 1.0e-3;
    static constexpr ScalarT betta = 2;
    static constexpr ScalarT kappa = 0;

    // Precomputed parameters
    static constexpr ScalarT lamda = alpha * alpha * (src_dim + kappa) - src_dim;
    static constexpr ScalarT n_lamda = src_dim + lamda;
    static constexpr ScalarT n_lamda_sqrt = sqrt(n_lamda);
    // first-order weights
    static constexpr ScalarT wm_0 = lamda / n_lamda; 
    static constexpr ScalarT wm_i = 0.5 / n_lamda; 
    // second-order weights
    static constexpr ScalarT wc_0 = wm_0 + (1 - alpha * alpha + betta); 
    static constexpr ScalarT wc_i = wm_i; 
    
    inline SrcPointT GeneratePoint(const SrcPointT& src_mean, const SrcCovarSqerT& src_cvr_sqrt, size_t idx) const {
        if (idx < src_dim) {
            return src_mean + src_cvr_sqrt.col(idx);
        } else {
            return src_mean - src_cvr_sqrt.col(idx - src_dim);
        }
    } 

public:
    DstDistT operator()(const FuncT& func, const SrcDistT& src) const {
        // generate update points and weights
        Eigen::Matrix<ScalarT, dst_dim, pnts_size> points;

        const SrcPointT scr_mean = src.mean();
        const SrcCovarSqerT scr_cvr_sqrt = n_lamda_sqrt * src.covar_sqrt();
        
        const auto ut_mean = func(scr_mean);
        for(size_t i = 0; i < pnts_size; ++i) {
            points.col(i) = func(GeneratePoint(scr_mean, scr_cvr_sqrt, i));
        }

        // approximate gausian destribution
        DstMeanT dst_mean = wm_0 * ut_mean;
        for(size_t i = 0; i < pnts_size; ++i) {
            dst_mean += wm_i * points.col(i);
        }

        const DstMeanT cntrd_mean = (ut_mean - dst_mean);
        DstCovarT dst_covar = wc_0 * cntrd_mean * cntrd_mean.transpose();
        
        for(size_t i = 0; i < pnts_size; ++i) {
            const DstMeanT cntrd_pnt = (points.col(i) - dst_mean);
            dst_covar += wc_i * cntrd_pnt * cntrd_pnt.transpose();
        }
        
        return DstDistT(dst_mean, dst_covar);
    } 
};

}
