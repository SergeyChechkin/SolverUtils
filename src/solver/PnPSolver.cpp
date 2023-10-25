#include "solver/PnPSolver.h"
#include "solver_utils/Transformation.h"
#include "solver_utils/PerspectiveProjection.h"
#include "solver_utils/Functions.h"

#include <Eigen/Eigenvalues>

#include <iostream>

using namespace solver;

PnPSolver::Report PnPSolver::SolvePose(
    const std::vector<Eigen::Vector3d>& points,
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
            GetPoseFactor(points[i], features[i], pose, J, error);

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
            std::cout << "mean cost - " << cost / factor_count << ", ";
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

void PnPSolver::GetPoseFactor(
    const Eigen::Vector3d& point,
    const Eigen::Vector2d& feature, 
    const Eigen::Vector<double, 6>& pose,
    Eigen::Matrix<double, 2, 6>& J,
    Eigen::Matrix<double, 2, 1>& error) 
{
    using namespace solver::transformation;
    using namespace solver::projection;

    const auto point_3d = IsometricTransformation<double>::f(pose, point);
    const auto projection = PerspectiveProjection<double>::f(point_3d);

    const auto point_3d_dps = IsometricTransformation<double>::df_dps(pose, point);
    const auto projection_dpt = PerspectiveProjection<double>::df_dpt(point_3d);
    J = projection_dpt * point_3d_dps;
    error = projection - feature;
}