#include "solver_utils/Rotation.h"

#include <ceres/jet.h>
#include <ceres/rotation.h>

#include <gtest/gtest.h>
#include <iostream>

using namespace solver::rotation;

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

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

