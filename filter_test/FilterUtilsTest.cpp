
#include <gaussian_filter/CovarianceMatrix.h>
#include <gaussian_filter/GaussianDistribution.h>
#include <gaussian_filter/GaussianUnscentedUpdate.h>

#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

#include <gtest/gtest.h>
#include <iostream>

using namespace gaussian_filter;

TEST(FilterUtils, CovatianceMatrixTest) { 

    Eigen::AngleAxisd aa(M_PI / 9, Eigen::Vector3d(1, 2, 3).normalized());
    Eigen::Vector3d diag(2, 1, 3);
    CovarianceMatrix<double, 3> cov_mat(diag, aa.matrix());
/*
    std::cout << std::endl;
    std::cout << "covariance matrix:" << std::endl;
    std::cout << cov_mat.GetCovarianceMatrix() << std::endl;
    std::cout << std::endl;
    std::cout << "sqrt test:" << std::endl;
    std::cout << cov_mat.GetCovarianceSqrt() * cov_mat.GetCovarianceSqrt().transpose() << std::endl;
    std::cout << std::endl;
    std::cout << "information matrix:" << std::endl;
    std::cout << cov_mat.GetInformationMatrix() << std::endl;
    std::cout << std::endl;
    std::cout << "inverse covariance matrix:" << std::endl;
    std::cout << cov_mat.GetCovarianceMatrix().inverse() << std::endl;
*/

    CovarianceMatrix<double, 3> cov_mat_(cov_mat.GetCovarianceMatrix());
    std::cout << std::endl;
    std::cout << cov_mat.GetCovarianceMatrix() << std::endl;
    std::cout << std::endl;
    std::cout << cov_mat_.GetCovarianceMatrix() << std::endl;
}

TEST(FilterUtils, InformationMatrixTest) { 
    std::clock_t cpu_start = std::clock();
    Eigen::Matrix<double, 6, 6> mat;
    for(long long i = 0; i < 1000000; ++i) {
        Eigen::Vector<double, 6> diag(1, 2, 3, 4, 5, 6);
        CovarianceMatrix<double, 6> cov_mat(diag, Eigen::Matrix<double, 6, 6>::Identity());

        mat = cov_mat.GetCovarianceMatrix();
    }
    float cpu_duration = 1000.0 * (std::clock() - cpu_start) / CLOCKS_PER_SEC;
    std::cout << "CPU time - " << cpu_duration << " ms." << std::endl;

    cpu_start = std::clock();
    for(long long i = 0; i < 1000000; ++i) {
        Eigen::Vector<double, 6> diag(1, 2, 3, 4, 5, 6);
        CovarianceMatrix<double, 6> cov_mat(diag, Eigen::Matrix<double, 6, 6>::Identity());

        mat = cov_mat.GetInformationMatrix();
    }
    cpu_duration = 1000.0 * (std::clock() - cpu_start) / CLOCKS_PER_SEC;
    std::cout << "CPU time - " << cpu_duration << " ms." << std::endl;
}

class PerspectiveProjection {
public:
    static constexpr size_t src_dim = 3; 
    static constexpr size_t dst_dim = 2; 
    using ScalarT = double;
    using SrcPointT = Eigen::Vector<ScalarT, src_dim>;
    using DstPointT = Eigen::Vector<ScalarT, dst_dim>;
public:
    DstPointT operator()(const SrcPointT& src) const {
        return {src[0]/src[2], src[0]/src[1]};
    } 
};

TEST(FilterUtils, UnscentedUpdateTest) {
    using InDistType = GaussianDistribution<double, 3>;
    InDistType::MeanT src_mean(1, 2, 3);
    InDistType::CovMatT src_covar;
    InDistType src_gss_dist(src_mean, src_covar);

    using FuncType = UpdateFunction<double, 3, 3>;
    UpdateFunction<double, 3, 3> lin_func;

    GaussianUnscentedUpdate<FuncType> guu;
    auto dst_gss_dist = guu(lin_func, src_gss_dist);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

