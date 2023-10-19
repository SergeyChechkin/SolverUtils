#include "solver_utils/Rotation.h"
#include "solver_utils/Transformation.h"

#include <ceres/jet.h>
#include <ceres/rotation.h>

#include <gtest/gtest.h>
#include <iostream>

using namespace solver::rotation;
using namespace solver::transformation;

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

TEST(SolverUtils, RotationTest) {
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

TEST(SolverUtils, TransformationTest) {
    const Eigen::Vector<double, 6> pose = {M_PI / 5, 0, 0, 1, 2, 3};
    const Eigen::Vector3d pt = {1, 1, 1};
    
    auto f = Transformation<double>::f(pose, pt);
    auto df_dps = Transformation<double>::df_dps(pose, pt);
    auto df_dpt = Transformation<double>::df_dpt(pose, pt);

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

    Eigen::Vector3<JetT> f_j = Transformation<JetT>::f(pose_j, pt_j);

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
    
    auto f = Transformation<double>::f(pose, pt);
    auto df_dps = Transformation<double>::df_dps_zero(pt);

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

    Eigen::Vector3<JetT> f_j = Transformation<JetT>::f(pose_j, pt_j);

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

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

