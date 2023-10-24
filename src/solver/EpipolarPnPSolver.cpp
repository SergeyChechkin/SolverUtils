#include "solver/EpipolarPnPSolver.h"
#include "solver_utils/Transformation.h"
#include "solver_utils/PerspectiveProjection.h"
#include "solver_utils/Functions.h"

#include <Eigen/Eigenvalues>

#include <ceres/jet.h>

#include <iostream>

using namespace solver;

EpipolarPnPSolver::Report EpipolarPnPSolver::SolvePose(
    const std::vector<Eigen::Vector3d>& points,
    const std::vector<Eigen::Vector2d>& points_info,
    const std::vector<Eigen::Vector2d>& features,  
    Eigen::Vector<double, 6>& pose, 
    const Cofiguration& config)
{
    const size_t factor_count = points.size();
    double last_mean_cost = 0;
    
    Eigen::Matrix<double, 6, 6> H;
    Eigen::Matrix<double, 6, 1> b;   

    Eigen::Matrix<double, 2, 6> J;
    Eigen::Matrix<double, 2, 1> error; 

    double factor_cost;  
    
    Report report;
    
    for(size_t itr = 0; itr < config.max_iterations; ++itr) {
        H = Eigen::Matrix<double, 6, 6>::Zero();
        b = Eigen::Matrix<double, 6, 1>::Zero();   
        double cost = 0;

        for(size_t i = 0; i < factor_count; ++i) {
            GetPoseFactor(points[i], points_info[i], features[i], pose, J, error);

            H += J.transpose() * J;
            b -= error.transpose() * J;
            cost += error.squaredNorm(); 
        }

        Eigen::Vector<double, 6> dx = H.ldlt().solve(b);
        pose += dx;
        
        double mean_cost = cost / factor_count;
        double cost_diff = last_mean_cost - mean_cost;
        double cost_change = std::abs(cost_diff) / mean_cost;

        if (config.verbal) {
            std::cout << "Iteration " << itr << ": "; 
            std::cout << "cost diff - " << cost_diff << ", ";
            std::cout << "mean cost - " << mean_cost << ", ";
            std::cout << "cost change - " << cost_change << ", ";
            std::cout << "step - " << dx.norm() << std::endl;
        }

        report.iterations = itr;
        report.cost = cost;
 
        if (mean_cost < config.min_cost)
            break;
        
        if (cost_change < config.min_cost_change)
            break;
        
        last_mean_cost = mean_cost;
    }

    return report;
}

void EpipolarPnPSolver::GetPoseFactor(
    const Eigen::Vector3d& point,
    const Eigen::Vector2d& point_info,
    const Eigen::Vector2d& feature, 
    const Eigen::Vector<double, 6>& pose,
    Eigen::Matrix<double, 2, 6>& J,
    Eigen::Matrix<double, 2, 1>& error) 
{
    using namespace solver::transformation;
    using namespace solver::projection;
    using JetT = ceres::Jet<double, 6>;
    using std::abs;

    Eigen::Vector<JetT, 6> pose_J;
    for(int i = 0; i < 6; ++i)
        pose_J[i] = JetT(pose[i], i);

    const auto point_3d_J = IsometricTransformation<JetT>::f(pose_J, point.cast<JetT>());
    const auto projection_J = PerspectiveProjection<JetT>::f(point_3d_J);
    const auto projection_error_J = projection_J - feature.cast<JetT>();

    const auto zero_depth_point_J = pose_J.tail(3);
    Eigen::Vector2<JetT> zero_depth_projection_J = Eigen::Vector2<JetT>::Zero();

    if (abs(zero_depth_point_J[2]) > std::numeric_limits<JetT>::epsilon()) {
        zero_depth_projection_J = PerspectiveProjection<JetT>::f(zero_depth_point_J);
    } 

    const Eigen::Vector2<JetT> epipolar_vector_J = (projection_J - zero_depth_projection_J).normalized();
    const Eigen::Vector2<JetT> epipolar_vector_tan_J(epipolar_vector_J[1], -epipolar_vector_J[0]);

    Eigen::Vector2<JetT> error_J;
    error_J[0] = JetT(point_info[0]) * epipolar_vector_J.dot(projection_error_J);
    error_J[1] = JetT(point_info[1]) * epipolar_vector_tan_J.dot(projection_error_J);

    error[0] = error_J[0].a;
    error[1] = error_J[1].a;
    J.row(0) = error_J[0].v;
    J.row(1) = error_J[1].v;
}