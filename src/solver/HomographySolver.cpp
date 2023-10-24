#include "solver/HomographySolver.h"
#include "solver_utils/Transformation.h"
#include "solver_utils/PerspectiveProjection.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <glog/logging.h>

#include <iostream>
#include <thread>

using namespace solver;

class Homography_aa_ADCF {
public:
    template<typename T>
    bool operator()(const T aa[3], const T t[3], T residuals[1]) const {

        Eigen::Matrix3<T> R;
        ceres::AngleAxisToRotationMatrix(aa, R.data());

        Eigen::Matrix3<T> t_x;
        t_x << T(0), t[2], -t[1], -t[2], T(0), t[0], t[1], -t[0], T(0);

        const Eigen::Matrix3<T> E = t_x * R;

        const Eigen::Vector3<T> pt_0 = plane_point_0_.homogeneous().cast<T>();
        const Eigen::Vector3<T> pt_1 = plane_point_1_.homogeneous().cast<T>();

        residuals[0] = pt_1.transpose() * E * pt_0;

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector2d& plane_point_0, const Eigen::Vector2d& plane_point_1) {
            return new ceres::AutoDiffCostFunction<Homography_aa_ADCF, 1, 3, 3>(
                new Homography_aa_ADCF(plane_point_0, plane_point_1));
        }

    Homography_aa_ADCF(const Eigen::Vector2d& plane_point_0, const Eigen::Vector2d& plane_point_1) 
        : plane_point_0_(plane_point_0), plane_point_1_(plane_point_1) {}
private:
    Eigen::Vector2d plane_point_0_;
    Eigen::Vector2d plane_point_1_;
};

bool HomographySolver::SolvePoseCeres(
    const std::vector<Eigen::Vector2d>& features_0,
    const std::vector<Eigen::Vector2d>& features_1, 
    Eigen::Vector<double, 6>& pose) 
{
    CHECK_EQ(features_0.size(), features_1.size());

    ceres::Problem problem;
    Eigen::Vector3d t(1, 1, 1);

    const double loss_threshold = 2.0 / 465;
    ceres::LossFunction* lf = new ceres::CauchyLoss(loss_threshold);

    for(size_t i = 0; i < features_0.size(); ++i) {
        auto* cf = Homography_aa_ADCF::Create(features_0[i], features_1[i]);
        problem.AddResidualBlock(cf, lf, pose.data(), t.data());
    }

    problem.SetManifold(t.data(), new ceres::SphereManifold<3>);

    ceres::Solver::Options options;
    //options.minimizer_progress_to_stdout = true;
    options.linear_solver_type = ceres::DENSE_QR;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    pose.tail(3) = t;

    //std::cerr << summary.BriefReport() << std::endl; 
    //std::cerr << summary.message << std::endl; 

    return summary.IsSolutionUsable();
    
}