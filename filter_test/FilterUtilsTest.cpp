
#include <gaussian_filter/CovarianceMatrix.h>
#include <gaussian_filter/GaussianDistribution.h>
#include <gaussian_filter/GaussianUnscentedUpdate.h>
#include <unscented_update/UpdateFunction.h>

#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

#include <gtest/gtest.h>
#include <iostream>

using namespace gaussian_filter;
using namespace unscented_update;

template<typename MatT>
void ASSERT_EQ_MAT(
    const MatT& mat_1, 
    const MatT& mat_2) {
        
    ASSERT_EQ(mat_1.rows(), mat_2.rows());
    ASSERT_EQ(mat_1.cols(), mat_2.cols());

    for(size_t i = 0; i < mat_1.rows(); ++i) {
        for(size_t j = 0; j < mat_1.cols(); ++j) {
            ASSERT_NEAR(mat_1(i,j), mat_2(i,j), 1e-5);
        }
    }
}

template<typename T, size_t Dm_N>
void ASSERT_EQ_NormDistr(
    const GaussianDistribution<T, Dm_N>& gd_1, 
    const GaussianDistribution<T, Dm_N>& gd_2) {

    ASSERT_EQ_MAT(gd_1.mean(), gd_2.mean()); 
    ASSERT_EQ_MAT(gd_1.covar(), gd_2.covar()); 
}


TEST(FilterUtils, CovatianceMatrixTest) { 

    Eigen::AngleAxisd aa(M_PI / 9, Eigen::Vector3d(1, 2, 3).normalized());
    Eigen::Vector3d diag(2, 1, 3);
    CovarianceMatrix<double, 3> cov_mat(diag, aa.matrix());
    CovarianceMatrix<double, 3> cov_mat_(cov_mat.GetCovarianceMatrix());

    ASSERT_EQ_MAT(cov_mat.GetCovarianceMatrix(), cov_mat_.GetCovarianceMatrix());
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
/*
TEST(FilterUtils, UnscentedUpdateTest) {
    using InDistType = GaussianDistribution<double, 3>;
    
    Eigen::AngleAxisd aa(M_PI / 9, Eigen::Vector3d(1, 2, 3).normalized());
    Eigen::Vector3d diag(2, 1, 3);
    InDistType::CovMatT src_covar(diag, aa.matrix());
    
    InDistType::MeanT src_mean(1, 2, 3);
    InDistType src_gss_dist(src_mean, src_covar);
    std::cout << src_gss_dist << std::endl;

    using FuncType = UpdateFunction<double, 3, 3>;
    UpdateFunction<double, 3, 3> lin_func(1);

    GaussianUnscentedUpdate<FuncType> guu;
    auto dst_gss_dist = guu(lin_func, src_gss_dist);

    ASSERT_EQ_NormDistr(src_gss_dist, dst_gss_dist);
}
*/
class PerspectiveProjection {
public:
    static constexpr size_t src_dim = 3; 
    static constexpr size_t dst_dim = 2; 
    using ScalarT = double;
    using SrcPointT = Eigen::Vector<ScalarT, src_dim>;
    using DstPointT = Eigen::Vector<ScalarT, dst_dim>;
public:
    DstPointT operator()(const SrcPointT& src) const {
        return {src[0]/src[2], src[1]/src[2]};
    } 
};

TEST(FilterUtils, UnscentedUpdateTest2) {
    using InDistType = GaussianDistribution<double, 3>;
    
    Eigen::AngleAxisd aa(M_PI / 9, Eigen::Vector3d(1, 2, 3).normalized());
    Eigen::Vector3d diag(1, 1, 1);
    InDistType::CovMatT src_covar(diag, aa.matrix());
    
    InDistType::MeanT src_mean(3, 2, 1);
    InDistType src_gss_dist(src_mean, src_covar);
    std::cout << src_gss_dist << std::endl;

    PerspectiveProjection func;

    GaussianUnscentedUpdate<PerspectiveProjection> guu;

    auto dst_gss_dist = guu(func, src_gss_dist);

    std::cout << std::endl;
    std::cout << dst_gss_dist << std::endl;
    std::cout << std::endl;
    std::cout << func(src_mean) << std::endl;
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

