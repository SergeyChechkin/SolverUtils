#pragma once

#include <Eigen/Core>
#include <type_traits>

// 1. Input distribution, generate points with weights  
// 2. Output distribution, approximate probability distribution from transformed points 
// 3. Update function, convert source points  
// 4. Unscented update, perform update 

/*
template<size_t Dm_N, size_t Pt_N>
class UpdatePoints {
public:
    Eigen::Matrix<double, Dm_N, Pt_N> points_;
    Eigen::Vector<double, Pt_N> weights_;
}; 
*/

template<typename DT>
class UpdateInput {
public:
    static constexpr size_t dim_size = Dm_N;// Input distrebution dimension 
    static constexpr size_t pnt_size = 0;   // Number of generated points
    using DistT = DT;                       // Update input destribution 
    using PointT = Eigen::Vector<double, DT::dim_size>;

    PointT GetPoint(size_t idx) const {
        return PointT::Zero();
    } 

    double GetWeight(size_t idx) const {
        return 1;
    }
};
/*
template<size_t src_Dm_N, size_t dst_Dm_N>
class UpdateFunction {
public:
    static constexpr size_t src_dim_size = src_Dm_N;// Input distrebution dimension 
    static constexpr size_t dst_dim_size = dst_Dm_N;// Output distrebution dimension 
    using SrcPointT = Eigen::Vector<double, src_Dm_N>;
    using DstPointT = Eigen::Vector<double, dst_Dm_N>;
public:
    DstPointT operator()(const SrcPointT& src) const {
        return DstPointT::Zero();
    } 
};

template<typename DistributionT, typename UpdateInputT, typename UpdateFunctionT>
class UpdateOutput {
public:
    using DistT = DistributionT;    // Update output destribution
    using PointsT = Eigen::Matrix<double, Dm_N, Pt_N>;
    using WeightsT = Eigen::Vector<double, Pt_N>;

    // approximate distribution
    static DistT Approximate(const PointsT& points, const WeightsT& weights) {
        DistT result;
        return result;
    }
};

template<typename InDistT, typename OutDistT, typename FuncT>
class UnscentedUpdate {
private:
    const InDistT& in_update_;
    const OutDistT& out_update_;
    const FuncT& func_;
public:
    UnscentedUpdate(
        const InDistT& in_update, 
        const OutDistT& out_update, 
        const FuncT& func) 
    : in_update_(in_update)
    , out_update_(out_update)
    , func_(func) {
    }
public:
    static OutDist::DistT Update(const InDist::DistT& src) {
        Eigen::Matrix<double, InDistT::dim_size, InDistT::pnt_size> points_;
        Eigen::Vector<double, OutDistT::dim_size> weights_;

        //Check weight types match in input and output distribution types. 
        static_assert(std::is_same<typename InDist::WeightsT, typename OutDist::WeightsT>::value == false, "Expect weights vectors of the same types.");
        
        // Generate points
        typename InDist::PointsT in_points;
        typename InDist::WeightT weights;
        InDist::Generate(in_points, weights);  

        // Transform points
        typename OutDist::PointsT out_points;
        for(size_t i = 0; i < InDist::Pt_N; ++i) {
            out_points.col(i) = Func::f(in_points.col(i));
        }

        // Approximate distribution
        return OutDist::Approximate(out_points, weights);
    }
};
*/