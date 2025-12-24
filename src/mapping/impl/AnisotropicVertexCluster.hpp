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

    /// Evaluates a consistent mapping and agglomerates the result in the given output data
    void mapConsistent(const time::Sample &inData, Eigen::VectorXd &outData) const;
    
    /// Number of input vertices this partition operates on
    unsigned int getNumberOfInputVertices() const;

    /// The center coordinate of this cluster
    std::array<double, 3> getCenterCoords() const;

    /// Invalidates and erases data structures the cluster holds
    void clear();

    /// Returns, whether the current cluster is empty or not, where empty means that there
    /// are either no input vertices or output vertices.
    bool empty() const;
    
    /// Set the normalized weight for the given \p vertexID in the outputMesh
    void setNormalizedWeight(double normalizedWeight, VertexID vertexID);

    /// Computes and saves the RBF coefficients
    void computeCacheData(const Eigen::Ref<const Eigen::MatrixXd> &globalIn, Eigen::MatrixXd &polyOut, Eigen::MatrixXd &coefficientsOut) const;

    void evaluateConservativeCache(Eigen::MatrixXd &epsilon, const Eigen::MatrixXd &Au, Eigen::Ref<Eigen::MatrixXd> out);


    Eigen::VectorXd interpolateAt(const mesh::Vertex &v, const Eigen::MatrixXd &poly, const Eigen::MatrixXd &coeffs, const mesh::Mesh &inMesh) const;

    void addWriteDataToCache(const mesh::Vertex &v, const Eigen::VectorXd &load, Eigen::MatrixXd &epsilon, Eigen::MatrixXd &Au,
                             const mesh::Mesh &inMesh);

    void initializeCacheData(Eigen::MatrixXd &polynomial, Eigen::MatrixXd &coeffs, const int nComponents);

    // ------------------------------- Others ----------------------------

    double computeWeight(const mesh::Vertex &v) const;

    void computeWeights(mesh::PtrMesh outMesh);

    Eigen::VectorXd getWeights() const;

    boost::container::flat_set<VertexID> getOutputIDs() const;

    // 判断顶点是否在椭球覆盖范围内，判据: (x-c)^T * M * (x-c) <= 1.0
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
    mesh::PtrMesh inMesh,
    mesh::PtrMesh outMesh,
    const PilotPoint& pilot, 
    double shapeParameter
) : _pilotID(pilot.id), _center(center), _radius(radius), _polynomial(polynomial), _function(function)
{
    // 1. Compute Shape Matrix and Transformation components
    PRECICE_ASSERT(radius > 0, "r must be positive");
    PRECICE_ASSERT(shapeParameter > 1, "shapeParameter must be greater than 1");
    computeShapeMatrix(pilot, radius, shapeParameter);

    // 2. Identify input and output vertex IDs within the anisotropic ellipsoidal region
    // We first get candidates within a bounding sphere, then filter based on Mahalanobis distance
    mesh::Mesh::VertexOffsets inIDs, outIDs;

    if (pilot.type == FeatureType::LINEAR) {
        inIDs = inMesh->index().getVerticesInsideBox(center, radius * shapeParameter);
        outIDs = outMesh->index().getVerticesInsideBox(center, radius * shapeParameter);
    } else {
        inIDs = inMesh->index().getVerticesInsideBox(center, radius);
        outIDs = outMesh->index().getVerticesInsideBox(center, radius);
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
    filterIDs(inIDs, inMesh);
    filterIDs(outIDs, outMesh);

    _inputIDs.insert(inIDs.begin(), inIDs.end());
    _outputIDs.insert(outIDs.begin(), outIDs.end());

    computeWeights(outMesh);

    // 5. Re-initialize RBF Solver with Transformed Data
    std::vector<bool> deadAxis(inMesh->getDimensions(), false);
    
    _rbfSolver = RadialBasisFctSolver<RADIAL_BASIS_FUNCTION_T>{
        function, 
        *inMesh.get(), 
        _inputIDs, 
        *outMesh.get(), 
        _outputIDs, 
        deadAxis, 
        _polynomial
    };
    
    _hasComputedMapping = true;
}

template <typename RADIAL_BASIS_FUNCTION_T>
void AnisotropicVertexCluster<RADIAL_BASIS_FUNCTION_T>::mapConservative(const time::Sample &inData, Eigen::VectorXd &outData) const
{
  // First, a few sanity checks. Empty partitions shouldn't be stored at all
  PRECICE_ASSERT(!empty());
  PRECICE_ASSERT(_hasComputedMapping);
  PRECICE_ASSERT(_normalizedWeights.size() == static_cast<Eigen::Index>(_outputIDs.size()));

  // Define an alias for data dimension in order to avoid ambiguity
  const unsigned int nComponents = inData.dataDims;
  const auto        &localInData = inData.values;

  // TODO: We can probably reduce the temporary allocations here
  Eigen::VectorXd in(_rbfSolver.getOutputSize());

  // Now we perform the data mapping component-wise
  for (unsigned int c = 0; c < nComponents; ++c) {
    // Step 1: extract the relevant input data from the global input data and store
    // it in a contiguous array, which is required for the RBF solver
    for (unsigned int i = 0; i < _outputIDs.size(); ++i) {
      const auto dataIndex = *(_outputIDs.nth(i));
      PRECICE_ASSERT(dataIndex * nComponents + c < localInData.size(), dataIndex * nComponents + c, localInData.size());
      PRECICE_ASSERT(_normalizedWeights[i] > 0, _normalizedWeights[i], i);
      // here, we also directly apply the weighting, i.e., we split the input data
      in[i] = localInData[dataIndex * nComponents + c] * _normalizedWeights[i];
    }

    // Step 2: solve the system using a conservative constraint
    auto result = _rbfSolver.solveConservative(in, _polynomial);
    PRECICE_ASSERT(result.size() == static_cast<Eigen::Index>(_inputIDs.size()));

    // Step 3: now accumulate the result into our global output data
    for (unsigned int i = 0; i < _inputIDs.size(); ++i) {
      const auto dataIndex = *(_inputIDs.nth(i));
      PRECICE_ASSERT(dataIndex * nComponents + c < outData.size(), dataIndex * nComponents + c, outData.size());
      outData[dataIndex * nComponents + c] += result(i);
    }
  }
}

template <typename RADIAL_BASIS_FUNCTION_T>
void AnisotropicVertexCluster<RADIAL_BASIS_FUNCTION_T>::mapConsistent(const time::Sample &inData, Eigen::VectorXd &outData) const
{
  // First, a few sanity checks. Empty partitions shouldn't be stored at all
  PRECICE_ASSERT(!empty());
  PRECICE_ASSERT(_hasComputedMapping);
  PRECICE_ASSERT(_normalizedWeights.size() == static_cast<Eigen::Index>(_outputIDs.size()));

  // Define an alias for data dimension in order to avoid ambiguity
  const unsigned int nComponents = inData.dataDims;
  const auto        &localInData = inData.values;

  Eigen::VectorXd in(_rbfSolver.getInputSize());

  // Now we perform the data mapping component-wise
  for (unsigned int c = 0; c < nComponents; ++c) {
    // Step 1: extract the relevant input data from the global input data and store
    // it in a contiguous array, which is required for the RBF solver (last polyparams entries remain zero)
    for (unsigned int i = 0; i < _inputIDs.size(); i++) {
      const auto dataIndex = *(_inputIDs.nth(i));
      PRECICE_ASSERT(dataIndex * nComponents + c < localInData.size(), dataIndex * nComponents + c, localInData.size());
      in[i] = localInData[dataIndex * nComponents + c];
    }

    // Step 2: solve the system using a consistent constraint
    auto result = _rbfSolver.solveConsistent(in, _polynomial);
    PRECICE_ASSERT(static_cast<Eigen::Index>(_outputIDs.size()) == result.size());

    // Step 3: now accumulate the result into our global output data
    for (unsigned int i = 0; i < _outputIDs.size(); ++i) {
      const auto dataIndex = *(_outputIDs.nth(i));
      PRECICE_ASSERT(dataIndex * nComponents + c < outData.size(), dataIndex * nComponents + c, outData.size());
      PRECICE_ASSERT(_normalizedWeights[i] > 0);
      // here, we also directly apply the weighting, i.e., split the result data
      outData[dataIndex * nComponents + c] += result(i) * _normalizedWeights[i];
    }
  }
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

template <typename RADIAL_BASIS_FUNCTION_T>
double AnisotropicVertexCluster<RADIAL_BASIS_FUNCTION_T>::computeWeight(const mesh::Vertex &v) const
{
    auto c = getCenterCoords();
    Eigen::Vector3d center(c[0], c[1], c[2]);

    Eigen::Vector3d vertex = v.getCoords();

    Eigen::Vector3d diff = vertex - center;
    const double d2 = diff.transpose() * _inverseCovariance * diff;

    if (d2 > 1.0 + math::NUMERICAL_ZERO_DIFFERENCE) {
        return 0.0;
    } else {
        return std::sqrt(d2);
    }
}

template <typename RADIAL_BASIS_FUNCTION_T>
void AnisotropicVertexCluster<RADIAL_BASIS_FUNCTION_T>::computeWeights(mesh::PtrMesh outMesh)
{   
    auto c = getCenterCoords();
    Eigen::Vector3d center(c[0], c[1], c[2]);

    for (std::size_t i = 0; i < _outputIDs.size(); ++i) {
        const auto dataIndex = *(_outputIDs.nth(i));
        PRECICE_ASSERT(dataIndex < _outputIDs.size(), dataIndex, _outputIDs.size());

        auto outVertex = outMesh->vertex(dataIndex);
        auto outVertexCoords = outVertex.rawCoords();
        Eigen::Vector3d outVextor(outVertexCoords[0], outVertexCoords[1], outVertexCoords[2]);
        Eigen::Vector3d diff = outVextor - center;

        const double d2 = diff.transpose() * _inverseCovariance * diff;

        if (d2 > 1.0 + math::NUMERICAL_ZERO_DIFFERENCE) {
            setNormalizedWeight(0.0, dataIndex);
        } else {
            setNormalizedWeight(std::sqrt(d2), dataIndex);
        }
    }
}

template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::VectorXd AnisotropicVertexCluster<RADIAL_BASIS_FUNCTION_T>::getWeights() const
{
    return _normalizedWeights;
}

template <typename RADIAL_BASIS_FUNCTION_T>
boost::container::flat_set<VertexID> AnisotropicVertexCluster<RADIAL_BASIS_FUNCTION_T>::getOutputIDs() const
{
    return _outputIDs;
}

template <typename RADIAL_BASIS_FUNCTION_T>
void AnisotropicVertexCluster<RADIAL_BASIS_FUNCTION_T>::setNormalizedWeight(double normalizedWeight, VertexID vertexID)
{
    PRECICE_ASSERT(_outputIDs.size() > 0);
    PRECICE_ASSERT(_outputIDs.contains(vertexID), vertexID);
    PRECICE_ASSERT(normalizedWeight >= 0.0, normalizedWeight);

    if (_normalizedWeights.size() == 0)
        _normalizedWeights.resize(_outputIDs.size());

    // The find method of boost flat_set comes with O(log(N)) complexity (the more expensive part here)
    auto localID = _outputIDs.index_of(_outputIDs.find(vertexID));

    PRECICE_ASSERT(static_cast<Eigen::Index>(localID) < _normalizedWeights.size(), localID, _normalizedWeights.size());
    _normalizedWeights[localID] = normalizedWeight;
}

template <typename RADIAL_BASIS_FUNCTION_T>
void AnisotropicVertexCluster<RADIAL_BASIS_FUNCTION_T>::evaluateConservativeCache(Eigen::MatrixXd &epsilon, const Eigen::MatrixXd &Au, Eigen::Ref<Eigen::MatrixXd> out)
{
  Eigen::MatrixXd localIn(_inputIDs.size(), Au.cols());
  _rbfSolver.evaluateConservativeCache(epsilon, Au, localIn);
  // Step 3: now accumulate the result into our global output data
  for (std::size_t i = 0; i < _inputIDs.size(); ++i) {
    const auto dataIndex = *(_inputIDs.nth(i));
    PRECICE_ASSERT(dataIndex < out.cols(), out.cols());
    out.col(dataIndex) += localIn.row(i);
  }
}

template <typename RADIAL_BASIS_FUNCTION_T>
void AnisotropicVertexCluster<RADIAL_BASIS_FUNCTION_T>::computeCacheData(const Eigen::Ref<const Eigen::MatrixXd> &globalIn, Eigen::MatrixXd &polyOut, Eigen::MatrixXd &coeffOut) const
{
  PRECICE_TRACE();
  Eigen::MatrixXd in(_rbfSolver.getInputSize(), globalIn.rows());
  // Step 1: extract the relevant input data from the global input data and store
  // it in a contiguous array, which is required for the RBF solver (last polyparams entries remain zero)
  for (std::size_t i = 0; i < _inputIDs.size(); i++) {
    const auto dataIndex = *(_inputIDs.nth(i));
    PRECICE_ASSERT(dataIndex < globalIn.cols(), globalIn.cols());
    in.row(i) = globalIn.col(dataIndex);
  }
  // Step 2: solve the system using a consistent constraint
  _rbfSolver.computeCacheData(in, _polynomial, polyOut, coeffOut);
}

template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::VectorXd AnisotropicVertexCluster<RADIAL_BASIS_FUNCTION_T>::interpolateAt(const mesh::Vertex &v, const Eigen::MatrixXd &poly, const Eigen::MatrixXd &coeffs, const mesh::Mesh &inMesh) const
{
  PRECICE_TRACE();
  return _rbfSolver.interpolateAt(v, poly, coeffs, _function, _inputIDs, inMesh);
}

template <typename RADIAL_BASIS_FUNCTION_T>
void AnisotropicVertexCluster<RADIAL_BASIS_FUNCTION_T>::addWriteDataToCache(const mesh::Vertex &v, const Eigen::VectorXd &load,
                                                                          Eigen::MatrixXd &epsilon, Eigen::MatrixXd &Au,
                                                                          const mesh::Mesh &inMesh)
{
  PRECICE_TRACE();
  _rbfSolver.addWriteDataToCache(v, load, epsilon, Au, _function, _inputIDs, inMesh);
}

template <typename RADIAL_BASIS_FUNCTION_T>
void AnisotropicVertexCluster<RADIAL_BASIS_FUNCTION_T>::initializeCacheData(Eigen::MatrixXd &poly, Eigen::MatrixXd &coeffs, const int nComponents)
{
  if (Polynomial::SEPARATE == _polynomial) {
    poly.resize(_rbfSolver.getNumberOfPolynomials(), nComponents);
  }
  coeffs.resize(_inputIDs.size(), nComponents);
}

} // namespace precice::mapping