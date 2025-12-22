#pragma once

#include <Eigen/Core>
#include <vector>
#include <Eigen/src/Eigenvalues/SelfAdjointEigenSolver.h>
#include <numeric>

// #include "mesh/Mesh.hpp"
#include "mapping/RadialBasisFctSolver.hpp"
#include "mapping/config/MappingConfiguration.hpp"
#include "mapping/impl/BasisFunctions.hpp"
#include "precice/impl/Types.hpp"
#include "profiling/Event.hpp"
#include "SphericalVertexCluster.hpp"

namespace precice::mapping {

using Vertices = std::vector<mesh::Vertex>;

// 导引点所属的局部特征类型，对应第一步"网格局部特征识别"的产出
enum class FeatureType {
    LINEAR,     // 线状 (Prolate)
    SURFACE,    // 面状 (Oblate)
    VOLUMETRIC  // 体状 (Sphere)
};

/**
 * @brief Storage the geometric feature.
 * @param position the physical coordinates of the feature(the position of the guide point)
 * @param eigenvalues the eigenvalues(lamda1-3) after PCA decomposition, used to determine the feature type
 * @param eigenvectors the eigenvector matrix(principal directions), 
 *        used to determine the rotational pose of the anisotropic ellipsoid
 * @param type the type of the feature, with enumeration options:{LINEAR, SURFACE, VOLUMETRIC}
*/
struct PilotPoint {
    Eigen::Vector3d position;
    FeatureType type;
    Eigen::Vector3d eigenvalues;  // lambda1 >= lambda2 >= lambda3
    Eigen::Matrix3d eigenvectors; // 列向量 v1, v2, v3
    int id;
};

template <typename RADIAL_BASIS_FUNCTION_T>
class AnisotropicVertexCluster : public SphericalVertexCluster<RADIAL_BASIS_FUNCTION_T> {
public:
    AnisotropicVertexCluster(
        mesh::Vertex center, 
        double radius,
        RADIAL_BASIS_FUNCTION_T function,
        Polynomial polynomial,
        mesh::PtrMesh inputMesh,
        mesh::PtrMesh outputMesh,
        const PilotPoint& pilot, 
        double shapeParameter = 3.0);

    /// Evaluates a conservative mapping and agglomerates the result in the given output data
    void mapConservative(const time::Sample &inData, Eigen::VectorXd &outData) const;

    /// Computes and saves the RBF coefficients
    void computeCacheData(const Eigen::Ref<const Eigen::MatrixXd> &globalIn, Eigen::MatrixXd &polyOut, Eigen::MatrixXd &coefficientsOut) const;

    /// Evaluates a consistent mapping and agglomerates the result in the given output data
    void mapConsistent(const time::Sample &inData, Eigen::VectorXd &outData) const;

    /// Set the normalized weight for the given \p vertexID in the outputMesh
    void setNormalizedWeight(double normalizedWeight, VertexID vertexID);

    void evaluateConservativeCache(Eigen::MatrixXd &epsilon, const Eigen::MatrixXd &Au, Eigen::Ref<Eigen::MatrixXd> out);

    /// Compute the weight for a given vertex
    double computeWeight(const mesh::Vertex &v) const;

    Eigen::VectorXd interpolateAt(const mesh::Vertex &v, const Eigen::MatrixXd &poly, const Eigen::MatrixXd &coeffs, const mesh::Mesh &inMesh) const;
    /// Number of input vertices this partition operates on
    unsigned int getNumberOfInputVertices() const;

    /// The center coordinate of this cluster
    std::array<double, 3> getCenterCoords() const;

    /// Invalidates and erases data structures the cluster holds
    void clear();

    /// Returns, whether the current cluster is empty or not, where empty means that there
    /// are either no input vertices or output vertices.
    bool empty() const;

    void addWriteDataToCache(const mesh::Vertex &v, const Eigen::VectorXd &load, Eigen::MatrixXd &epsilon, Eigen::MatrixXd &Au,
                            const mesh::Mesh &inMesh);

    void initializeCacheData(Eigen::MatrixXd &polynomial, Eigen::MatrixXd &coeffs, const int nComponents);

    // 判断顶点是否在椭球覆盖范围内
    // 判据: (x-c)^T * M * (x-c) <= 1.0
    bool isCovering(const Eigen::Vector3d& vertex) const;

    void computeShapeMatrix(const PilotPoint& pilot, double r, double alpha);

private:
    int _pilotID;

    // 运行时缓存的逆协方差矩阵，用于快速判断覆盖 (Mahalanobis distance)
    Eigen::Matrix3d _inverseCovariance;
};

// --------------------------------------------------- HEADER IMPLEMENTATIONS

template <typename RADIAL_BASIS_FUNCTION_T>
AnisotropicVertexCluster<RADIAL_BASIS_FUNCTION_T>::AnisotropicVertexCluster(
    mesh::Vertex center, 
    double radius,
    RADIAL_BASIS_FUNCTION_T function,
    Polynomial polynomial,
    mesh::PtrMesh inputMesh,
    mesh::PtrMesh outputMesh,
    const PilotPoint& pilot, 
    double shapeParameter
) : SphericalVertexCluster<RADIAL_BASIS_FUNCTION_T>(
    center, 
    radius,
    function,
    polynomial,
    inputMesh,
    outputMesh,
    )
    : _pilotID(pilot.id)
{
    PRECICE_ASSERT(radius > 0, "r must be positive");
    PRECICE_ASSERT(shapeParameter > 0 && shapeParameter < 1, "shapeParameter must be (0, 1)");
    computeShapeMatrix(pilot, radius, shapeParameter);
}


template <typename RADIAL_BASIS_FUNCTION_T>
bool AnisotropicVertexCluster<RADIAL_BASIS_FUNCTION_T>::isCovering(const Eigen::Vector3d& vertex) const 
{
    Eigen::Vector3d diff = vertex - _center;
    return (diff.transpose() * _inverseCovariance * diff).value() <= 1.0;
}

template <typename RADIAL_BASIS_FUNCTION_T>
void AnisotropicVertexCluster<RADIAL_BASIS_FUNCTION_T>::computeShapeMatrix(const PilotPoint& pilot, double r, double alpha) {
    // r: 基础半径 (由k近邻距离决定)
    // alpha: 经验缩放系数 (注意事项4中的"固定经验值")
    
    Eigen::Vector3d semiAxes;
    Eigen::Matrix3d rotation = pilot.eigenvectors; // v1, v2, v3

    // 注意事项 4 & 5 的实现逻辑：
    switch (pilot.type) {
        case FeatureType::VOLUMETRIC: // 圆球
            semiAxes = Eigen::Vector3d(r, r, r);
            rotation = Eigen::Matrix3d::Identity(); // 无方向
            break;

        case FeatureType::LINEAR: // 长球 (Prolate)
            // 长轴(v1)放大，短轴(v2, v3)为基础半径
            semiAxes = Eigen::Vector3d(r / alpha, r, r);
            break;

        case FeatureType::SURFACE: // 扁球 (Oblate)
            // 法向v3压缩，切向v1/v2保持
            semiAxes = Eigen::Vector3d(r, r, r * alpha); 
            break;
    }

    // 构建形状矩阵 M = R * S^(-2) * R^T
    // 使得 d^2 = x^T M x <= 1
    Eigen::Matrix3d S_inv2 = Eigen::Matrix3d::Zero();
    S_inv2(0, 0) = 1.0 / (semiAxes[0] * semiAxes[0]);
    S_inv2(1, 1) = 1.0 / (semiAxes[1] * semiAxes[1]);
    S_inv2(2, 2) = 1.0 / (semiAxes[2] * semiAxes[2]);

    _inverseCovariance = rotation * S_inv2 * rotation.transpose();
}


} // namespace precice::mapping::impl