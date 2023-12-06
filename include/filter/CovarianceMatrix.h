#pragma once

#include <Eigen/Core>

template<typename T, size_t Sz>
class CovarianceMatrix {
public:
    using MatrixType = Eigen::Matrix<T, Sz, Sz>;
    using DiagonalType = Eigen::Vector<T, Sz>;

    CovarianceMatrix() 
    : dgnl_(DiagonalType::Ones())
    , rot_(MatrixType::Identity()) {
    }

    CovarianceMatrix(const DiagonalType& diagonal, const MatrixType& rotation) 
    : dgnl_(diagonal)
    , rot_(rotation) {
    }

    // returns covariance matrix
    MatrixType GetCovarianceMatrix() const {
        return rot_ * dgnl_.asDiagonal() * rot_.transpose();
    }

    // returns information matrix
    // inverse diagonal only
    MatrixType GetInformationMatrix() const {
        return rot_ * dgnl_.cwiseInverse().asDiagonal() * rot_.transpose();
    }

private:
    DiagonalType dgnl_; // principal components
    MatrixType rot_;    // rotation matrix 
};