#include "solver/PnPSolver.h"
#include "solver/BASolver.h"
#include "solver/HomographySolver.h"
#include "solver/EpipolarPnPSolver.h"

#include "solver_utils/Rotation.h"
#include "solver_utils/Transformation.h"
#include "solver_utils/PerspectiveProjection.h"

#include <ceres/jet.h>
#include <ceres/rotation.h>

#include <gtest/gtest.h>
#include <random>
#include <iostream>

using namespace solver;
using namespace solver::rotation;
using namespace solver::transformation;
using namespace solver::projection;

TEST(SolverUtils, RotationMinTest) { 

    const Eigen::Vector3d aa = {0.00001, 0, 0};
    const Eigen::Vector3d pt = {1, 1, 1};
    
    auto f = Rotation_min<double>::f(aa, pt);
    auto df_daa = Rotation_min<double>::df_daa(pt);
    auto df_dpt = Rotation_min<double>::df_dpt(aa);

    // using ceres::Jet for diravatives validation 
    using JetT = ceres::Jet<double, 6>;
    Eigen::Vector3<JetT> pt_j, aa_j;
    for(int i = 0; i < 3; ++i)  {
        aa_j[i] = JetT(aa[i], i);
        pt_j[i] = JetT(pt[i], i+3);
    }

    auto f_j = Rotation_min<JetT>::f(aa_j, pt_j);

    ASSERT_DOUBLE_EQ(f_j[0].a, f[0]);
    ASSERT_DOUBLE_EQ(f_j[1].a, f[1]);
    ASSERT_DOUBLE_EQ(f_j[2].a, f[2]);

    ASSERT_DOUBLE_EQ(f_j[0].v[0], df_daa(0, 0));
    ASSERT_DOUBLE_EQ(f_j[1].v[0], df_daa(1, 0));
    ASSERT_DOUBLE_EQ(f_j[2].v[0], df_daa(2, 0));

    ASSERT_DOUBLE_EQ(f_j[0].v[1], df_daa(0, 1));
    ASSERT_DOUBLE_EQ(f_j[1].v[1], df_daa(1, 1));
    ASSERT_DOUBLE_EQ(f_j[2].v[1], df_daa(2, 1));

    ASSERT_DOUBLE_EQ(f_j[0].v[2], df_daa(0, 2));
    ASSERT_DOUBLE_EQ(f_j[1].v[2], df_daa(1, 2));
    ASSERT_DOUBLE_EQ(f_j[2].v[2], df_daa(2, 2));

    ASSERT_DOUBLE_EQ(f_j[0].v[3], df_dpt(0, 0));
    ASSERT_DOUBLE_EQ(f_j[1].v[3], df_dpt(1, 0));
    ASSERT_DOUBLE_EQ(f_j[2].v[3], df_dpt(2, 0));

    ASSERT_DOUBLE_EQ(f_j[0].v[4], df_dpt(0, 1));
    ASSERT_DOUBLE_EQ(f_j[1].v[4], df_dpt(1, 1));
    ASSERT_DOUBLE_EQ(f_j[2].v[4], df_dpt(2, 1));

    ASSERT_DOUBLE_EQ(f_j[0].v[5], df_dpt(0, 2));
    ASSERT_DOUBLE_EQ(f_j[1].v[5], df_dpt(1, 2));
    ASSERT_DOUBLE_EQ(f_j[2].v[5], df_dpt(2, 2));   
}

TEST(SolverUtils, RotationClassTest) {
    const Eigen::Vector3d aa = {0.1, 0, 0};
    const Eigen::Vector3d pt = {1, 2, 3};

    auto f = Rotation<double>::f(aa, pt);
    auto df_daa = Rotation<double>::df_daa(aa, pt);
    auto df_dpt = Rotation<double>::df_dpt(aa, pt);

    // using ceres::Jet for diravatives validation 
    using JetT = ceres::Jet<double, 6>;
    Eigen::Vector3<JetT> pt_j, aa_j;
    for(int i = 0; i < 3; ++i)  {
        aa_j[i] = JetT(aa[i], i);
        pt_j[i] = JetT(pt[i], i+3);
    }

    auto f_j = Rotation<JetT>::f(aa_j, pt_j);

    ASSERT_NEAR(f_j[0].a, f[0], FLT_EPSILON);
    ASSERT_NEAR(f_j[1].a, f[1], FLT_EPSILON);
    ASSERT_NEAR(f_j[2].a, f[2], FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[0], df_daa(0, 0), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[0], df_daa(1, 0), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[0], df_daa(2, 0), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[1], df_daa(0, 1), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[1], df_daa(1, 1), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[1], df_daa(2, 1), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[2], df_daa(0, 2), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[2], df_daa(1, 2), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[2], df_daa(2, 2), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[3], df_dpt(0, 0), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[3], df_dpt(1, 0), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[3], df_dpt(2, 0), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[4], df_dpt(0, 1), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[4], df_dpt(1, 1), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[4], df_dpt(2, 1), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[5], df_dpt(0, 2), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[5], df_dpt(1, 2), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[5], df_dpt(2, 2), FLT_EPSILON);
}

TEST(SolverUtils, RotationObjectTest) {
    const Eigen::Vector3d aa = {0.1, 0, 0};
    const Eigen::Vector3d pt = {1, 2, 3};

    Rotation<double> rot(aa);

    auto f = rot.f(pt);
    auto df_daa = rot.df_daa(pt);
    auto df_dpt = rot.df_dpt(pt);

    // using ceres::Jet for diravatives validation 
    using JetT = ceres::Jet<double, 6>;
    Eigen::Vector3<JetT> pt_j, aa_j;
    for(int i = 0; i < 3; ++i)  {
        aa_j[i] = JetT(aa[i], i);
        pt_j[i] = JetT(pt[i], i+3);
    }

    auto f_j = Rotation<JetT>::f(aa_j, pt_j);

    ASSERT_NEAR(f_j[0].a, f[0], FLT_EPSILON);
    ASSERT_NEAR(f_j[1].a, f[1], FLT_EPSILON);
    ASSERT_NEAR(f_j[2].a, f[2], FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[0], df_daa(0, 0), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[0], df_daa(1, 0), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[0], df_daa(2, 0), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[1], df_daa(0, 1), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[1], df_daa(1, 1), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[1], df_daa(2, 1), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[2], df_daa(0, 2), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[2], df_daa(1, 2), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[2], df_daa(2, 2), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[3], df_dpt(0, 0), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[3], df_dpt(1, 0), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[3], df_dpt(2, 0), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[4], df_dpt(0, 1), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[4], df_dpt(1, 1), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[4], df_dpt(2, 1), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[5], df_dpt(0, 2), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[5], df_dpt(1, 2), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[5], df_dpt(2, 2), FLT_EPSILON);
}

TEST(SolverUtils, TransformationClassTest) {
    const Eigen::Vector<double, 6> pose = {M_PI / 5, 0, 0, 1, 2, 3};
    const Eigen::Vector3d pt = {1, 1, 1};
    
    auto f = IsometricTransformation<double>::f(pose, pt);
    auto df_dps = IsometricTransformation<double>::df_dps(pose, pt);
    auto df_dpt = IsometricTransformation<double>::df_dpt(pose, pt);

    // using ceres::Jet for comparosing 
    using JetT = ceres::Jet<double, 9>;
    Eigen::Vector<JetT, 6> pose_j;
    Eigen::Vector3<JetT> pt_j;
    for(int i = 0; i < 6; ++i)  {
        pose_j[i] = JetT(pose[i], i);
    }
    for(int i = 0; i < 3; ++i)  {
        pt_j[i] = JetT(pt[i], i + 6);
    }

    Eigen::Vector3<JetT> f_j = IsometricTransformation<JetT>::f(pose_j, pt_j);

    ASSERT_NEAR(f_j[0].a, f[0], FLT_EPSILON);
    ASSERT_NEAR(f_j[1].a, f[1], FLT_EPSILON);
    ASSERT_NEAR(f_j[2].a, f[2], FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[0], df_dps(0, 0), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[0], df_dps(1, 0), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[0], df_dps(2, 0), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[1], df_dps(0, 1), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[1], df_dps(1, 1), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[1], df_dps(2, 1), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[2], df_dps(0, 2), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[2], df_dps(1, 2), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[2], df_dps(2, 2), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[3], df_dps(0, 3), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[3], df_dps(1, 3), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[3], df_dps(2, 3), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[4], df_dps(0, 4), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[4], df_dps(1, 4), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[4], df_dps(2, 4), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[5], df_dps(0, 5), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[5], df_dps(1, 5), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[5], df_dps(2, 5), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[6], df_dpt(0, 0), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[6], df_dpt(1, 0), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[6], df_dpt(2, 0), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[7], df_dpt(0, 1), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[7], df_dpt(1, 1), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[7], df_dpt(2, 1), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[8], df_dpt(0, 2), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[8], df_dpt(1, 2), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[8], df_dpt(2, 2), FLT_EPSILON);
}

TEST(SolverUtils, TransformationObjectTest) {
    const Eigen::Vector<double, 6> pose = {M_PI / 5, 0, 0, 1, 2, 3};
    const Eigen::Vector3d pt = {1, 1, 1};
    
    IsometricTransformation<double> it(pose);

    auto f = it.f(pose, pt);
    auto df_dps = it.df_dps(pose, pt);
    auto df_dpt = it.df_dpt(pose, pt);

    // using ceres::Jet for comparosing 
    using JetT = ceres::Jet<double, 9>;
    Eigen::Vector<JetT, 6> pose_j;
    Eigen::Vector3<JetT> pt_j;
    for(int i = 0; i < 6; ++i)  {
        pose_j[i] = JetT(pose[i], i);
    }
    for(int i = 0; i < 3; ++i)  {
        pt_j[i] = JetT(pt[i], i + 6);
    }

    Eigen::Vector3<JetT> f_j = IsometricTransformation<JetT>::f(pose_j, pt_j);

    ASSERT_NEAR(f_j[0].a, f[0], FLT_EPSILON);
    ASSERT_NEAR(f_j[1].a, f[1], FLT_EPSILON);
    ASSERT_NEAR(f_j[2].a, f[2], FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[0], df_dps(0, 0), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[0], df_dps(1, 0), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[0], df_dps(2, 0), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[1], df_dps(0, 1), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[1], df_dps(1, 1), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[1], df_dps(2, 1), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[2], df_dps(0, 2), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[2], df_dps(1, 2), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[2], df_dps(2, 2), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[3], df_dps(0, 3), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[3], df_dps(1, 3), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[3], df_dps(2, 3), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[4], df_dps(0, 4), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[4], df_dps(1, 4), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[4], df_dps(2, 4), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[5], df_dps(0, 5), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[5], df_dps(1, 5), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[5], df_dps(2, 5), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[6], df_dpt(0, 0), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[6], df_dpt(1, 0), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[6], df_dpt(2, 0), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[7], df_dpt(0, 1), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[7], df_dpt(1, 1), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[7], df_dpt(2, 1), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[8], df_dpt(0, 2), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[8], df_dpt(1, 2), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[8], df_dpt(2, 2), FLT_EPSILON);
}

TEST(SolverUtils, ZeroTransformationTest) {
    const Eigen::Vector<double, 6> pose = {0, 0, 0, 0, 0, 0};
    const Eigen::Vector3d pt = {1, 2, 3};
    
    auto f = IsometricTransformation<double>::f(pose, pt);
    auto df_dps = IsometricTransformation<double>::df_dps_zero(pt);

    // using ceres::Jet for comparosing 
    using JetT = ceres::Jet<double, 9>;
    Eigen::Vector<JetT, 6> pose_j;
    Eigen::Vector3<JetT> pt_j;
    for(int i = 0; i < 6; ++i)  {
        pose_j[i] = JetT(pose[i], i);
    }
    for(int i = 0; i < 3; ++i)  {
        pt_j[i] = JetT(pt[i], i + 6);
    }

    Eigen::Vector3<JetT> f_j = IsometricTransformation<JetT>::f(pose_j, pt_j);

    ASSERT_NEAR(f_j[0].a, f[0], FLT_EPSILON);
    ASSERT_NEAR(f_j[1].a, f[1], FLT_EPSILON);
    ASSERT_NEAR(f_j[2].a, f[2], FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[0], df_dps(0, 0), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[0], df_dps(1, 0), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[0], df_dps(2, 0), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[1], df_dps(0, 1), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[1], df_dps(1, 1), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[1], df_dps(2, 1), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[2], df_dps(0, 2), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[2], df_dps(1, 2), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[2], df_dps(2, 2), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[3], df_dps(0, 3), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[3], df_dps(1, 3), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[3], df_dps(2, 3), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[4], df_dps(0, 4), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[4], df_dps(1, 4), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[4], df_dps(2, 4), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[5], df_dps(0, 5), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[5], df_dps(1, 5), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[5], df_dps(2, 5), FLT_EPSILON);
}


TEST(SolverUtils, PerspectiveProjectionTest) {
    const Eigen::Vector3d pt = {1, 2, 3};
    const auto f = PerspectiveProjection<double>::f(pt);
    const auto df_dpt = PerspectiveProjection<double>::df_dpt(pt); 

    //std::cout << f.transpose() << std::endl;
    //std::cout << df_dpt << std::endl;


    // using ceres::Jet for comparosing 
    using JetT = ceres::Jet<double, 3>;
    Eigen::Vector3<JetT> pt_j;
    for(int i = 0; i < 3; ++i)  {
        pt_j[i] = JetT(pt[i], i);
    }

    Eigen::Vector2<JetT> f_j = PerspectiveProjection<JetT>::f(pt_j);

    //std::cout << f_j << std::endl;

    ASSERT_NEAR(f_j[0].a, f[0], FLT_EPSILON);
    ASSERT_NEAR(f_j[1].a, f[1], FLT_EPSILON);
    
    ASSERT_NEAR(f_j[0].v[0], df_dpt(0, 0), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[0], df_dpt(1, 0), FLT_EPSILON);
    
    ASSERT_NEAR(f_j[0].v[1], df_dpt(0, 1), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[1], df_dpt(1, 1), FLT_EPSILON);
    
    ASSERT_NEAR(f_j[0].v[2], df_dpt(0, 2), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[2], df_dpt(1, 2), FLT_EPSILON);
}

TEST(SolverUtils, PerspectiveProjectionCamTest) {
    const Eigen::Vector3d pt = {1, 2, 3};
    const Eigen::Vector3d cam = {450, 320, 240};    
    const auto f = PerspectiveProjection<double>::f(cam, pt);
    const auto df_cam = PerspectiveProjection<double>::df_dcm(cam, pt);
    const auto df_dpt = PerspectiveProjection<double>::df_dpt(cam, pt); 

    using JetT = ceres::Jet<double, 6>;
    Eigen::Vector3<JetT> cam_j;
    for(int i = 0; i < 3; ++i)  {
        cam_j[i] = JetT(cam[i], i);
    }
    Eigen::Vector3<JetT> pt_j;
    for(int i = 0; i < 3; ++i)  {
        pt_j[i] = JetT(pt[i], i+3);
    }

    Eigen::Vector2<JetT> f_j = PerspectiveProjection<JetT>::f(cam_j, pt_j);

    ASSERT_NEAR(f_j[0].a, f[0], FLT_EPSILON);
    ASSERT_NEAR(f_j[1].a, f[1], FLT_EPSILON);
    
    ASSERT_NEAR(f_j[0].v[3], df_dpt(0, 0), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[3], df_dpt(1, 0), FLT_EPSILON);
    
    ASSERT_NEAR(f_j[0].v[4], df_dpt(0, 1), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[4], df_dpt(1, 1), FLT_EPSILON);
    
    ASSERT_NEAR(f_j[0].v[5], df_dpt(0, 2), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[5], df_dpt(1, 2), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[0], df_cam(0, 0), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[0], df_cam(1, 0), FLT_EPSILON);
    
    ASSERT_NEAR(f_j[0].v[1], df_cam(0, 1), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[1], df_cam(1, 1), FLT_EPSILON);
    
    ASSERT_NEAR(f_j[0].v[2], df_cam(0, 2), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[2], df_cam(1, 2), FLT_EPSILON);
}

TEST(SolverUtils, PerspectiveRerojectionTest) {
    const Eigen::Vector2d pt = {2, 3};
    double inv_depth = 0.2;
    const Eigen::Vector3d cam = {450, 320, 240};    
    const auto f = PerspectiveReprojectionUnitPlane<double>::f(cam, pt, inv_depth);
    const auto df_dpt = PerspectiveReprojectionUnitPlane<double>::df_dpt(cam, pt, inv_depth); 
    const auto df_did = PerspectiveReprojectionUnitPlane<double>::df_did(cam, pt, inv_depth); 
    const auto df_dcm = PerspectiveReprojectionUnitPlane<double>::df_dcm(cam, pt, inv_depth); 

    using JetT = ceres::Jet<double, 6>;
    Eigen::Vector3<JetT> cam_j;
    for(int i = 0; i < 3; ++i)  {
        cam_j[i] = JetT(cam[i], i);
    }
    Eigen::Vector2<JetT> pt_j;
    for(int i = 0; i < 2; ++i)  {
        pt_j[i] = JetT(pt[i], i + 3);
    }
    JetT inv_depth_j(inv_depth, 5); 

    auto f_j = PerspectiveReprojectionUnitPlane<JetT>::f(cam_j, pt_j, inv_depth_j);

    ASSERT_NEAR(f_j[0].a, f[0], FLT_EPSILON);
    ASSERT_NEAR(f_j[1].a, f[1], FLT_EPSILON);
    ASSERT_NEAR(f_j[2].a, f[2], FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[0], df_dcm(0, 0), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[0], df_dcm(1, 0), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[0], df_dcm(2, 0), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[1], df_dcm(0, 1), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[1], df_dcm(1, 1), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[1], df_dcm(2, 1), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[2], df_dcm(0, 2), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[2], df_dcm(1, 2), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[2], df_dcm(2, 2), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[3], df_dpt(0, 0), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[3], df_dpt(1, 0), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[3], df_dpt(2, 0), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[4], df_dpt(0, 1), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[4], df_dpt(1, 1), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[4], df_dpt(2, 1), FLT_EPSILON);

    ASSERT_NEAR(f_j[0].v[5], df_did(0, 0), FLT_EPSILON);
    ASSERT_NEAR(f_j[1].v[5], df_did(1, 0), FLT_EPSILON);
    ASSERT_NEAR(f_j[2].v[5], df_did(2, 0), FLT_EPSILON);
}

TEST(SolverUtils, PnPTest) {

    const Eigen::Vector<double, 6> pose = {M_PI / 10, 0, 0, 1, 2, 3};
    const Eigen::Vector3d pnt = {-3, -2, -1};

    const auto point_3d = IsometricTransformation<double>::f(pose, pnt);
    const auto projection = PerspectiveProjection<double>::f(point_3d);

    const auto point_3d_dps = IsometricTransformation<double>::df_dps(pose, pnt);
    const auto projection_dpt = PerspectiveProjection<double>::df_dpt(point_3d);
    const auto projection_dps = projection_dpt * point_3d_dps;

    //std::cout << point_3d.transpose() << std::endl;
    //std::cout << point_3d_dps << std::endl;
    //std::cout << projection_dpt << std::endl;
    //std::cout << projection_dps << std::endl;

    using JetT = ceres::Jet<double, 6>;
    Eigen::Vector<JetT, 6> pose_j;
    Eigen::Vector3<JetT> pnt_j;
    for(int i = 0; i < 6; ++i)  {
        pose_j[i] = JetT(pose[i], i);
    }
    for(int i = 0; i < 3; ++i)  {
        pnt_j[i] = JetT(pnt[i]);
    }

    const auto point_3d_j = IsometricTransformation<JetT>::f(pose_j, pnt_j);
    const auto projection_j = PerspectiveProjection<JetT>::f(point_3d_j); 

    //std::cout << projection_j << std::endl;
}

TEST(SolverUtils, PnPSolverTest) { 
    std::default_random_engine gen;
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Eigen::Vector<double, 6> pose = {0.1, 0.2, 0.3, 0.1, 0.2, 0.3};
    
    size_t size = 1000;
    std::vector<Eigen::Vector3d> map(size);
    std::vector<Eigen::Vector2d> frame_t(size);

    for(size_t i = 0; i < size; ++i) {
        const Eigen::Vector3d point_3d(dist(gen), dist(gen), dist(gen) + 2);
        const Eigen::Vector3d point_3d_t = IsometricTransformation<double>::f(pose, point_3d);
        map[i] = point_3d;
        frame_t[i] = Eigen::Vector2d(point_3d_t[0]/point_3d_t[2], point_3d_t[1]/point_3d_t[2]);
    }    

    Eigen::Vector<double, 6> slvd_pose;
    slvd_pose.setZero();
    
    PnPSolver::Cofiguration pnp_config;
    pnp_config.verbal = false;
    
    std::clock_t cpu_start = std::clock();
    PnPSolver::SolvePose(map, frame_t, slvd_pose, pnp_config);
    float cpu_duration = 1000.0 * (std::clock() - cpu_start) / CLOCKS_PER_SEC;
    std::cout << "CPU time - " << cpu_duration << " ms." << std::endl;
    
    for(int i = 0; i < 6; ++i)
        ASSERT_NEAR(pose[i], slvd_pose[i], 1.0e-3);
}

TEST(SolverUtils, HomographySolverTest) { 
    std::default_random_engine gen;
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Eigen::Vector<double, 6> pose = {0.1, 0.2, 0.3, 0.1, 0.2, 0.3};
    
    size_t size = 1000;
    std::vector<Eigen::Vector3d> map(size);
    std::vector<Eigen::Vector2d> frame_0(size);
    std::vector<Eigen::Vector2d> frame_1(size);

    for(size_t i = 0; i < size; ++i) {
        const Eigen::Vector3d point_3d(dist(gen), dist(gen), dist(gen) + 2);
        const Eigen::Vector3d point_3d_t = IsometricTransformation<double>::f(pose, point_3d);
        map[i] = point_3d;
        frame_0[i] = PerspectiveProjection<double>::f(point_3d);
        frame_1[i] = PerspectiveProjection<double>::f(point_3d_t);
    }    

    Eigen::Vector<double, 6> slvd_pose;
    slvd_pose.setZero();
    
    std::clock_t cpu_start = std::clock();
    HomographySolver::SolvePoseCeres(frame_0, frame_1, slvd_pose);
    float cpu_duration = 1000.0 * (std::clock() - cpu_start) / CLOCKS_PER_SEC;
    std::cout << "CPU time - " << cpu_duration << " ms." << std::endl;

    //std::cout << slvd_pose.transpose() << std::endl;
}

TEST(SolverUtils, EpipolarPnPSolver) { 
    std::default_random_engine gen;
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Eigen::Vector<double, 6> pose = {0.1, 0.2, 0.3, 0.1, 0.2, 0.3};
    
    size_t size = 1000;
    std::vector<Eigen::Vector3d> map(size);
    std::vector<Eigen::Vector2d> frame_0(size);
    std::vector<Eigen::Vector2d> frame_1(size);

    std::vector<Eigen::Vector3d> slvd_map(size);
    std::vector<Eigen::Vector2d> info_map(size, {0.0001, 1});

    for(size_t i = 0; i < size; ++i) {
        const Eigen::Vector3d point_3d(dist(gen), dist(gen), dist(gen) + 2);
        const Eigen::Vector3d point_3d_t = IsometricTransformation<double>::f(pose, point_3d);
        map[i] = point_3d;
        frame_0[i] = PerspectiveProjection<double>::f(point_3d);
        frame_1[i] = PerspectiveProjection<double>::f(point_3d_t);
        slvd_map[i] = point_3d.normalized();
    }

    Eigen::Vector<double, 6> slvd_pose;
    slvd_pose.setZero();
    
    EpipolarPnPSolver::Cofiguration pnp_config;
    pnp_config.verbal = false;

    std::clock_t cpu_start = std::clock();
    EpipolarPnPSolver::SolvePose(slvd_map, info_map, frame_1, slvd_pose, pnp_config);
    float cpu_duration = 1000.0 * (std::clock() - cpu_start) / CLOCKS_PER_SEC;
    std::cout << "CPU time - " << cpu_duration << " ms." << std::endl;

    std::cout << slvd_pose.transpose() << std::endl;    
}

TEST(SolverUtils, BASolverTest) { 
    std::default_random_engine gen;
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Eigen::Vector<double, 6> pose = {0.1, 0.2, 0.3, 0.1, 0.2, 0.3};
    //Eigen::Vector<double, 6> pose = {0.1, 0.0, 0.0, 0.0, 0.0, 0.0};
    
    size_t size = 1000;
    std::vector<Eigen::Vector3d> map(size);
    std::vector<Eigen::Vector3d> slvd_map(size);
    std::vector<Eigen::Vector2d> frame_0(size);
    std::vector<Eigen::Vector2d> frame_1(size);

    for(size_t i = 0; i < size; ++i) {
        const Eigen::Vector3d point_3d(dist(gen), dist(gen), dist(gen) + 2);
        const Eigen::Vector3d point_3d_t = IsometricTransformation<double>::f(pose, point_3d);
        map[i] = point_3d;
        slvd_map[i] = point_3d.normalized();
        //slvd_map[i] = point_3d;
        frame_0[i] = PerspectiveProjection<double>::f(point_3d);
        frame_1[i] = PerspectiveProjection<double>::f(point_3d_t);
    }    

    Eigen::Vector<double, 6> slvd_pose;
    slvd_pose.setZero();
    
    std::clock_t cpu_start = std::clock();
    BASolver::SolvePosePointsCeres(frame_0, frame_1, slvd_pose, slvd_map);
    //BASolver::SolvePosePoints(frame_0, frame_1, slvd_pose, slvd_map);
    float cpu_duration = 1000.0 * (std::clock() - cpu_start) / CLOCKS_PER_SEC;
    std::cout << "CPU time - " << cpu_duration << " ms." << std::endl;

    std::cout << "BASolver: "  << slvd_pose.transpose() << std::endl;
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

