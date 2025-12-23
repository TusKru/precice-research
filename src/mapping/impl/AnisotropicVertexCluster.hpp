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
class AnisotropicVertexCluster {
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
    // void computeCacheData(const Eigen::Ref<const Eigen::MatrixXd> &globalIn, Eigen::MatrixXd &polyOut, Eigen::MatrixXd &coefficientsOut) const;

    /// Evaluates a consistent mapping and agglomerates the result in the given output data
    void mapConsistent(const time::Sample &inData, Eigen::VectorXd &outData) const;

    /// Set the normalized weight for the given \p vertexID in the outputMesh
    // void setNormalizedWeight(double normalizedWeight, VertexID vertexID);

    // void evaluateConservativeCache(Eigen::MatrixXd &epsilon, const Eigen::MatrixXd &Au, Eigen::Ref<Eigen::MatrixXd> out);

    /// Compute the weight for a given vertex
    double computeWeight(const mesh::Vertex &v) const;

    // Eigen::VectorXd interpolateAt(const mesh::Vertex &v, const Eigen::MatrixXd &poly, const Eigen::MatrixXd &coeffs, const mesh::Mesh &inMesh) const;
    
    /// Number of input vertices this partition operates on
    unsigned int getNumberOfInputVertices() const;

    /// The center coordinate of this cluster
    std::array<double, 3> getCenterCoords() const;

    /// Invalidates and erases data structures the cluster holds
    void clear();

    /// Returns, whether the current cluster is empty or not, where empty means that there
    /// are either no input vertices or output vertices.
    bool empty() const;

    // void addWriteDataToCache(const mesh::Vertex &v, const Eigen::VectorXd &load, Eigen::MatrixXd &epsilon, Eigen::MatrixXd &Au,
    //                         const mesh::Mesh &inMesh);

    // void initializeCacheData(Eigen::MatrixXd &polynomial, Eigen::MatrixXd &coeffs, const int nComponents);

    // ------------------------------- Others ----------------------------

    // 判断顶点是否在椭球覆盖范围内
    // 判据: (x-c)^T * M * (x-c) <= 1.0
    bool isCovering(const Eigen::Vector3d& vertex) const;

    void computeShapeMatrix(const PilotPoint& pilot, double r, double alpha);

private:
    /// logger, as usual
    mutable precice::logging::Logger _log{"mapping::SphericalVertexCluster"};

    /// center vertex of the cluster
    mesh::Vertex _center;

    /// radius of the vertex cluster
    const double _radius;

    /// The RBF solver
    RadialBasisFctSolver<RADIAL_BASIS_FUNCTION_T> _rbfSolver;

    // Stores the global IDs of the vertices so that we can apply a binary
    // search in order to query specific objects. Here we have logarithmic
    // complexity for setting the weights (only performed once), and constant
    // complexity when traversing through the IDs (performed in each iteration).

    /// Global VertexIDs associated to the input mesh (see constructor)
    boost::container::flat_set<VertexID> _inputIDs;

    /// Global VertexIDs associated to the output mesh (see constructor)
    boost::container::flat_set<VertexID> _outputIDs;

    /// Vector containing the normalized weights used for the output mesh data
    /// (consistent mapping) or input mesh data (conservative data)
    Eigen::VectorXd _normalizedWeights;

    /// Polynomial treatment in the RBF solver
    Polynomial _polynomial;

    RADIAL_BASIS_FUNCTION_T _function;

    /// The weighting function
    CompactPolynomialC2 _weightingFunction;

    /// Boolean switch in order to indicate that a mapping was computed
    bool _hasComputedMapping = false;

    // ------------------------------- Others ----------------------------
    
    int _pilotID;

    // 运行时缓存的逆协方差矩阵，用于快速判断覆盖 (Mahalanobis distance)
    Eigen::Matrix3d _inverseCovariance;

    // Transformation components for mapping to unit sphere
    Eigen::Matrix3d _rotation; // Eigenvectors
    Eigen::Vector3d _inverseSemiAxes; // 1/semiAxes
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
) : _pilotID(pilot.id), _center(center), _radius(radius), _polynomial(polynomial), _function(function), _weightingFunction(radius)
{
    // 1. Compute Shape Matrix and Transformation components
    PRECICE_ASSERT(radius > 0, "r must be positive");
    PRECICE_ASSERT(shapeParameter > 1, "shapeParameter must be greater than 1");
    computeShapeMatrix(pilot, radius, shapeParameter);

    // 2. Identify input and output vertex IDs within the anisotropic ellipsoidal region
    // We first get candidates within a bounding sphere, then filter based on Mahalanobis distance
    if (pilot.type == FeatureType::LINEAR) {
        auto inIDs = inputMesh->index().getVerticesInsideBox(center, radius * shapeParameter);
        auto outIDs = outputMesh->index().getVerticesInsideBox(center, radius * shapeParameter);
    } else {
        auto inIDs = inputMesh->index().getVerticesInsideBox(center, radius);
        auto outIDs = outputMesh->index().getVerticesInsideBox(center, radius);
    }

    auto filterIDs = [&](mesh::Mesh::VertexOffsets& ids, mesh::PtrMesh mesh) {
        for (auto it = ids.begin(); it != ids.end(); ) {
            const auto& v = mesh->vertex(*it);
            Eigen::Vector3d vPos;
            if (v.getDimensions() == 3) {
                vPos << v.coord(0), v.coord(1), v.coord(2);
            } else {
                vPos << v.coord(0), v.coord(1), 0.0;
            }

            if (!isCovering(vPos)) {
                it = ids.erase(it);
            } else {
                ++it;
            }
        }
    };
    // Access protected members from SphericalVertexCluster
    filterIDs(inIDs, inputMesh);
    filterIDs(outIDs, outputMesh);

    _inputIDs.insert(inIDs.begin(), inIDs.end());
    _outputIDs.insert(outIDs.begin(), outIDs.end());

    // 5. Re-initialize RBF Solver with Transformed Data
    std::vector<bool> deadAxis(inputMesh.getDimensions(), false);
    
    _rbfSolver = RadialBasisFctSolver<RADIAL_BASIS_FUNCTION_T>{
        function, 
        *inputMesh.get(), 
        _inputIDs, 
        *outputMesh.get(), 
        _outputIDs, 
        deadAxis, 
        _polynomial
    };
    
    _hasComputedMapping = true;
}


template <typename RADIAL_BASIS_FUNCTION_T>
bool AnisotropicVertexCluster<RADIAL_BASIS_FUNCTION_T>::isCovering(const Eigen::Vector3d& vertex) const 
{
    auto c = this->getCenterCoords();
    Eigen::Vector3d center(c[0], c[1], c[2]);
    Eigen::Vector3d diff = vertex - center;
    return (diff.transpose() * _inverseCovariance * diff).value() <= 1.0;
}

template <typename RADIAL_BASIS_FUNCTION_T>
void AnisotropicVertexCluster<RADIAL_BASIS_FUNCTION_T>::computeShapeMatrix(const PilotPoint& pilot, double r, double alpha) 
{
    // r: 基础半径 (由k近邻距离决定)
    // alpha: 经验缩放系数
    
    Eigen::Vector3d semiAxes;
    _rotation = pilot.eigenvectors; // v1, v2, v3

    switch (pilot.type) {
        case FeatureType::VOLUMETRIC: // 圆球
            semiAxes = Eigen::Vector3d(r, r, r);
            _rotation = Eigen::Matrix3d::Identity(); // 无方向
            break;

        case FeatureType::LINEAR: // 长球 (Prolate)
            // 长轴(v1)放大，短轴(v2, v3)为基础半径
            semiAxes = Eigen::Vector3d(r * alpha, r, r);
            break;

        case FeatureType::SURFACE: // 扁球 (Oblate)
            // 法向v3压缩，切向v1/v2保持
            semiAxes = Eigen::Vector3d(r, r, r / alpha); 
            break;
    }

    _inverseSemiAxes << 1.0/semiAxes[0], 1.0/semiAxes[1], 1.0/semiAxes[2];

    // M = R * S^(-2) * R^T
    Eigen::Matrix3d S_inv2 = Eigen::Matrix3d::Zero();
    S_inv2(0, 0) = _inverseSemiAxes[0] * _inverseSemiAxes[0];
    S_inv2(1, 1) = _inverseSemiAxes[1] * _inverseSemiAxes[1];
    S_inv2(2, 2) = _inverseSemiAxes[2] * _inverseSemiAxes[2];

    // d^2 = x^T M x
    _inverseCovariance = _rotation * S_inv2 * _rotation.transpose();
}

template <typename RADIAL_BASIS_FUNCTION_T>
bool AnisotropicVertexCluster<RADIAL_BASIS_FUNCTION_T>::empty() const 
{
    return _inputIDs.size() == 0;
}

template <typename RADIAL_BASIS_FUNCTION_T>
void AnisotropicVertexCluster<RADIAL_BASIS_FUNCTION_T>::clear() 
{
    PRECICE_TRACE();
    _inputIDs.clear();
    _outputIDs.clear();
    _rbfSolver.clear();
    _hasComputedMapping = false;
}

template <typename RADIAL_BASIS_FUNCTION_T>
std::array<double, 3> AnisotropicVertexCluster<RADIAL_BASIS_FUNCTION_T>::getCenterCoords() const 
{
    return _center.rawCoords();
}

template <typename RADIAL_BASIS_FUNCTION_T>
unsigned int AnisotropicVertexCluster<RADIAL_BASIS_FUNCTION_T>::getNumberOfInputVertices() const 
{
    return _inputIDs.size();
}

// @todo
template <typename RADIAL_BASIS_FUNCTION_T>
void AnisotropicVertexCluster<RADIAL_BASIS_FUNCTION_T>::mapConservative(const time::Sample &inData, Eigen::VectorXd &outData) const
{
    SphericalVertexCluster<RADIAL_BASIS_FUNCTION_T>::mapConservative(inData, outData);
}

template <typename RADIAL_BASIS_FUNCTION_T>
void AnisotropicVertexCluster<RADIAL_BASIS_FUNCTION_T>::mapConsistent(const time::Sample &inData, Eigen::VectorXd &outData) const
{
    SphericalVertexCluster<RADIAL_BASIS_FUNCTION_T>::mapConsistent(inData, outData);
}

template <typename RADIAL_BASIS_FUNCTION_T>
double AnisotropicVertexCluster<RADIAL_BASIS_FUNCTION_T>::computeWeight(const mesh::Vertex &v) const
{
    // auto c = this->getCenterCoords();
    // Eigen::Vector3d center(c[0], c[1], c[2]);
    // auto rv = v.rawCoords();
    // Eigen::Vector3d pv(rv[0], rv[1], rv[2]);
    // Eigen::Vector3d diff = pv - center;
    // const double d2 = diff.transpose() * _inverseCovariance * diff;
    // const double d  = std::sqrt(std::max(d2, 0.0));
    // CompactPolynomialC2 wf(_radius);
    // return wf.evaluate(d);

    Eigen::Vector3d centerVec(this->_center.coord(0), this->_center.coord(1), this->_center.coord(2));
    Eigen::Vector3d vVec;
    if (v.getDimensions() == 3) {
        vVec << v.coord(0), v.coord(1), v.coord(2);
    } else {
        vVec << v.coord(0), v.coord(1), 0.0;
    }
    
    Eigen::Vector3d diff = vVec - centerVec;
    double distSq = (diff.transpose() * _inverseCovariance * diff).value();
    
    if (distSq > 1.0 + math::NUMERICAL_ZERO_DIFFERENCE) {
        return 0.0;
    }
    
    // Clamp to 1.0 to avoid numerical issues with sqrt(>1) if slightly over
    if (distSq > 1.0) distSq = 1.0;

    // Use the updated weighting function (radius 1.0)
    return this->_weightingFunction.evaluate(std::sqrt(distSq));
}

} // namespace precice::mapping