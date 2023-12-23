#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>


namespace gaussian_filter {

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

    CovarianceMatrix(const MatrixType& covar) {
        // SVD decomposition
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, Sz, Sz>> eigen_solver(covar);
        if(Eigen::Success != eigen_solver.info()) {
            // TODO: handle exception
        }

        dgnl_ = eigen_solver.eigenvalues();
        rot_ = eigen_solver.eigenvectors();
    }

    // returns covariance matrix
    MatrixType GetCovarianceMatrix() const {
        return rot_ * dgnl_.asDiagonal() * rot_.transpose();
    }

    // returns information matrix
    MatrixType GetInformationMatrix() const {
        return rot_ * dgnl_.cwiseInverse().asDiagonal() * rot_.transpose();
    }

    // returns square root of covariance matrix
    MatrixType GetCovarianceSqrt() const {
        return rot_ * dgnl_.cwiseSqrt().asDiagonal();
    }

private:
    DiagonalType dgnl_; // principal components
    MatrixType rot_;    // rotation matrix 
};

}