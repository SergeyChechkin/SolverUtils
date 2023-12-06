
#include <filter/CovarianceMatrix.h>

#include <Eigen/Geometry>

#include <gtest/gtest.h>
#include <iostream>

TEST(FilterUtils, CovatianceMatrixTest) { 

    Eigen::AngleAxisd aa(M_PI / 9, Eigen::Vector3d(1, 2, 3).normalized());
    Eigen::Vector3d diag(1, 2, 3);
    CovarianceMatrix<double, 3> cov_mat(diag, aa.matrix());

    std::cout << std::endl;
    std::cout << "covariance matrix:" << std::endl;
    std::cout << cov_mat.GetCovarianceMatrix() << std::endl;
    std::cout << std::endl;
    std::cout << "information matrix:" << std::endl;
    std::cout << cov_mat.GetInformationMatrix() << std::endl;
    std::cout << std::endl;
    std::cout << "inverse covariance matrix:" << std::endl;
    std::cout << cov_mat.GetCovarianceMatrix().inverse() << std::endl;
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

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

