#include "solver/BASolver.h"
#include "solver_utils/Transformation.h"
#include "solver_utils/PerspectiveProjection.h"

#include <ceres/ceres.h>
#include <glog/logging.h>

#include <iostream>
#include <thread>

using namespace solver;

class BA_ADCF {
public:
    template<typename T>
    bool operator()(const T pose[6], const T pnt[3], T residuals[2]) const {
        const Eigen::Vector<T, 6> pose_(pose);
        const Eigen::Vector<T, 3> pnt_(pnt);

        const auto pnt_t = solver::transformation::IsometricTransformation<T>::f(pose_, pnt_);
        const auto projection = solver::projection::PerspectiveProjection<T>::f(pnt_t);

        residuals[0] = T(up_point_[0]) - projection[0];
        residuals[1] = T(up_point_[1]) - projection[1];

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector2d& up_point) {
            return new ceres::AutoDiffCostFunction<BA_ADCF, 2, 6, 3>(
                new BA_ADCF(up_point));
        }

    BA_ADCF(const Eigen::Vector2d& up_point) 
        : up_point_(up_point) {}
private:
    Eigen::Vector2d up_point_;
};

class BA_FIXED_ADCF {
public:
    template<typename T>
    bool operator()(const T pnt[3], T residuals[2]) const {
        const Eigen::Vector<T, 3> pnt_(pnt);

        const auto projection = solver::projection::PerspectiveProjection<T>::f(pnt_);

        residuals[0] = T(up_point_[0]) - projection[0];
        residuals[1] = T(up_point_[1]) - projection[1];

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector2d& up_point) {
            return new ceres::AutoDiffCostFunction<BA_FIXED_ADCF, 2, 3>(
                new BA_FIXED_ADCF(up_point));
        }

    BA_FIXED_ADCF(const Eigen::Vector2d& up_point) 
        : up_point_(up_point) {}
private:
    Eigen::Vector2d up_point_;
};

class BA_CF : public ceres::SizedCostFunction<2, 6, 3> {
public:
    public:
        bool Evaluate(double const* const* parameters,
            double* residuals,
            double** jacobians) const override 
        {
            Eigen::Map<const Eigen::Vector<double, 6>> pose(parameters[0]);
            Eigen::Map<const Eigen::Vector<double, 3>> pnt(parameters[1]);

            const auto pnt_t = solver::transformation::IsometricTransformation<double>::f(pose, pnt);
            const auto projection = solver::projection::PerspectiveProjection<double>::f(pnt_t);

            Eigen::Map<Eigen::Vector2<double>> res(residuals);
            res = projection - up_point_;

            if (jacobians) {
                const auto projection_dpt = solver::projection::PerspectiveProjection<double>::df_dpt(pnt_t);
                if (jacobians[0]) {
                    Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J0(jacobians[0]);
                    const auto point_3d_dps = solver::transformation::IsometricTransformation<double>::df_dps(pose, pnt);
                    J0 = projection_dpt * point_3d_dps;
                }

                if (jacobians[1]) {
                    Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J1(jacobians[1]);
                    const auto point_3d_dpt = solver::transformation::IsometricTransformation<double>::df_dpt(pose, pnt);
                    J1 = projection_dpt * point_3d_dpt;
                }
            }

            return true;
        }

    static ceres::CostFunction* Create(const Eigen::Vector2d& up_point) {
        return new BA_CF(up_point);
    }
private:
    BA_CF(const Eigen::Vector2d& up_point) 
        : up_point_(up_point) {}
private:
    Eigen::Vector2d up_point_;
};



class BA_FIXED_CF : public ceres::SizedCostFunction<2, 3> {
public:
    bool Evaluate(double const* const* parameters,
            double* residuals,
            double** jacobians) const override 
        {
            Eigen::Map<const Eigen::Vector<double, 3>> pnt(parameters[0]);
            const auto projection = solver::projection::PerspectiveProjection<double>::f(pnt);

            Eigen::Map<Eigen::Vector2<double>> res(residuals);
            res = projection - up_point_;

            if (jacobians) {
                if (jacobians[0]) {
                    Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J0(jacobians[0]);
                    const auto projection_dpt = solver::projection::PerspectiveProjection<double>::df_dpt(pnt);
                    J0 = projection_dpt;
                }
            }

            return true;
        }

    static ceres::CostFunction* Create(const Eigen::Vector2d& up_point) {
        return new BA_FIXED_CF(up_point);
    }
private:
    BA_FIXED_CF(const Eigen::Vector2d& up_point) 
        : up_point_(up_point) {}
private:
    Eigen::Vector2d up_point_;
};

void BASolver::SolvePosePointsCeres(
    const std::vector<Eigen::Vector2d>& features_0,
    const std::vector<Eigen::Vector2d>& features_1, 
    Eigen::Vector<double, 6>& pose, 
    std::vector<Eigen::Vector3d>& points) {
        
        CHECK_EQ(features_0.size(), features_1.size());
        ceres::Problem problem;

        const double loss_threshold = 2.0 / 465;    // about 2 pixels threshold
        ceres::LossFunction* lf = new ceres::CauchyLoss(loss_threshold);
        //ceres::LossFunction* lf = nullptr;

        size_t size = features_0.size();
        points.resize(size);

        for(int i = 0; i < size; ++i) {
            //problem.AddResidualBlock(BA_FIXED_ADCF::Create(features_0[i]), lf, points[i].data());
            //problem.AddResidualBlock(BA_ADCF::Create(features_1[i]), lf, pose.data(), points[i].data());

            problem.AddResidualBlock(BA_FIXED_CF::Create(features_0[i]), lf, points[i].data());
            problem.AddResidualBlock(BA_CF::Create(features_1[i]), lf, pose.data(), points[i].data());
        }

        ceres::Solver::Options options;
        //options.minimizer_progress_to_stdout = true;
        //options.num_threads = std::thread::hardware_concurrency();
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
    }