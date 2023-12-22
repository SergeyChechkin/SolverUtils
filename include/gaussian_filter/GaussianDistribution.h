#pragma once

#include "gaussian_filter/CovarianceMatrix.h"
#include <Eigen/Core>

namespace gaussian_filter {

template<typename T, size_t Dm_N>
class GaussianDistribution {
public:
    static constexpr size_t dim_size = Dm_N;
    using ScalatT = T;
    using MeanT = Eigen::Vector<T, dim_size>;
    using CovMatT = CovarianceMatrix<T, dim_size>;
    using MatT = typename CovMatT::MatrixType;
public:
    GaussianDistribution() 
    : mean_(MeanT::Zero()) {
    }

    GaussianDistribution(
        const MeanT& mean, 
        const CovMatT& covar) 
    : mean_(mean)
    , covar_(covar) {
    }

    GaussianDistribution(
        const MeanT& mean, 
        const MatT& covar) 
    : mean_(mean)
    , covar_(covar) {
    }

    const MeanT& mean() const {return mean_;}
    MatT covar() const {return covar_.GetCovarianceMatrix();}
    MatT covar_sqrt() const {return covar_.GetCovarianceSqrt();}
    MatT info() const {return covar_.GetInformationMatrix();}
private:
    MeanT mean_;
    CovMatT covar_;
};

}

