#include "solver/EpipolarPnPSolver.h"
#include "solver_utils/Transformation.h"
#include "solver_utils/PerspectiveProjection.h"
#include "solver_utils/Functions.h"

#include <Eigen/Eigenvalues>

#include <ceres/jet.h>

#include <iostream>

using namespace solver;
using namespace solver::transformation;
using namespace solver::projection;

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

        IsometricTransformation<double> it(pose);

        for(size_t i = 0; i < factor_count; ++i) {
            GetPoseFactor(points[i], points_info[i], features[i], it, J, error);

            H += J.transpose() * J;
            b -= error.transpose() * J;
            cost += error.squaredNorm(); 
        }
        // solve linear system
        Eigen::Vector<double, 6> dx = H.ldlt().solve(b);
        // update pose
        pose += dx;

        double mean_cost = cost / factor_count;
        double cost_diff = last_mean_cost - mean_cost;
        double cost_change = std::abs(cost_diff) / mean_cost;

        if (config.verbal) {
            std::cout << "Iteration " << itr << ": "; 
            std::cout << "mean cost - " << mean_cost << ", ";
            std::cout << "cost diff - " << cost_diff << ", ";
            std::cout << "cost change - " << cost_change << ", ";
            std::cout << "step - " << dx.norm() << std::endl;
        }

        report.iterations = itr;
        report.min_cost = cost;

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
    using JetT = ceres::Jet<double, 6>;
    using std::abs;

    const auto point_trans = IsometricTransformation<double>::f(pose, point);
    const auto projection = PerspectiveProjection<double>::f(point_trans);
    const auto projection_error = projection - feature;

    const auto point_trans_dps = IsometricTransformation<double>::df_dps(pose, point);
    const auto projection_dpt = PerspectiveProjection<double>::df_dpt(point_trans);
    const auto projection_dps = projection_dpt * point_trans_dps;

    const auto zero_depth_point = pose.tail(3);

    Eigen::Vector2d zero_depth_projection = Eigen::Vector2d::Zero();
    Eigen::Matrix<double, 2, 6> zero_depth_projection_dps = Eigen::Matrix<double, 2, 6>::Zero();

    if (abs(zero_depth_point[2]) > std::numeric_limits<double>::epsilon()) {
        zero_depth_projection = PerspectiveProjection<double>::f(zero_depth_point);

        Eigen::Matrix<double, 3, 6> zero_depth_point_dps = Eigen::Matrix<double, 3, 6>::Zero();
        zero_depth_point_dps.block<3,3>(0, 3) = Eigen::Matrix3d::Identity();
        zero_depth_projection_dps = PerspectiveProjection<double>::df_dpt(zero_depth_point) * zero_depth_point_dps;
    } 

    //TODO: derive the rest 
    Eigen::Vector<JetT, 2> projection_J;
    projection_J[0].a = projection[0];
    projection_J[1].a = projection[1];
    projection_J[0].v = projection_dps.row(0);
    projection_J[1].v = projection_dps.row(1);
    
    Eigen::Vector<JetT, 2> zero_depth_projection_J;
    zero_depth_projection_J[0].a = zero_depth_projection[0];
    zero_depth_projection_J[1].a = zero_depth_projection[1];
    zero_depth_projection_J[0].v = zero_depth_projection_dps.row(0);
    zero_depth_projection_J[1].v = zero_depth_projection_dps.row(1);

    const Eigen::Vector2<JetT> epipolar_vector_J = (projection_J - zero_depth_projection_J).normalized();
    const Eigen::Vector2<JetT> epipolar_vector_tan_J(epipolar_vector_J[1], -epipolar_vector_J[0]);

    const auto projection_error_J = projection_J - feature.cast<JetT>();

    Eigen::Vector2<JetT> error_J;
    error_J[0] = JetT(point_info[0]) * epipolar_vector_J.dot(projection_error_J);
    error_J[1] = JetT(point_info[1]) * epipolar_vector_tan_J.dot(projection_error_J);

    error[0] = error_J[0].a;
    error[1] = error_J[1].a;
    J.row(0) = error_J[0].v;
    J.row(1) = error_J[1].v;
}

void EpipolarPnPSolver::GetPoseFactor(
    const Eigen::Vector3d& point,
    const Eigen::Vector2d& point_info,
    const Eigen::Vector2d& feature, 
    const transformation::IsometricTransformation<double>& pose,
    Eigen::Matrix<double, 2, 6>& J,
    Eigen::Matrix<double, 2, 1>& error) 
{
    using JetT = ceres::Jet<double, 6>;
    using std::abs;

    const auto point_trans = pose.f(point);
    const auto projection = PerspectiveProjection<double>::f(point_trans);
    const auto projection_error = projection - feature;

    const auto point_trans_dps = pose.df_dps(point);
    const auto projection_dpt = PerspectiveProjection<double>::df_dpt(point_trans);
    const auto projection_dps = projection_dpt * point_trans_dps;

    const auto zero_depth_point = pose.translation();

    Eigen::Vector2d zero_depth_projection = Eigen::Vector2d::Zero();
    Eigen::Matrix<double, 2, 6> zero_depth_projection_dps = Eigen::Matrix<double, 2, 6>::Zero();

    if (abs(zero_depth_point[2]) > std::numeric_limits<double>::epsilon()) {
        zero_depth_projection = PerspectiveProjection<double>::f(zero_depth_point);

        Eigen::Matrix<double, 3, 6> zero_depth_point_dps = Eigen::Matrix<double, 3, 6>::Zero();
        zero_depth_point_dps.block<3,3>(0, 3) = Eigen::Matrix3d::Identity();
        zero_depth_projection_dps = PerspectiveProjection<double>::df_dpt(zero_depth_point) * zero_depth_point_dps;
    } 

    //TODO: derive the rest 
    Eigen::Vector<JetT, 2> projection_J;
    projection_J[0].a = projection[0];
    projection_J[1].a = projection[1];
    projection_J[0].v = projection_dps.row(0);
    projection_J[1].v = projection_dps.row(1);
    
    Eigen::Vector<JetT, 2> zero_depth_projection_J;
    zero_depth_projection_J[0].a = zero_depth_projection[0];
    zero_depth_projection_J[1].a = zero_depth_projection[1];
    zero_depth_projection_J[0].v = zero_depth_projection_dps.row(0);
    zero_depth_projection_J[1].v = zero_depth_projection_dps.row(1);

    const Eigen::Vector2<JetT> epipolar_vector_J = (projection_J - zero_depth_projection_J).normalized();
    const Eigen::Vector2<JetT> epipolar_vector_tan_J(epipolar_vector_J[1], -epipolar_vector_J[0]);

    const auto projection_error_J = projection_J - feature.cast<JetT>();

    Eigen::Vector2<JetT> error_J;
    error_J[0] = JetT(point_info[0]) * epipolar_vector_J.dot(projection_error_J);
    error_J[1] = JetT(point_info[1]) * epipolar_vector_tan_J.dot(projection_error_J);

    error[0] = error_J[0].a;
    error[1] = error_J[1].a;
    J.row(0) = error_J[0].v;
    J.row(1) = error_J[1].v;
}