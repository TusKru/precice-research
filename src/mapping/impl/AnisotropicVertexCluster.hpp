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

struct GlobalAnisotropyParams {
  Eigen::Matrix3d rotation;
  Eigen::Vector3d semiAxes;
  Eigen::Matrix3d inverseCovariance;
  double coverSearchRadius;
};

template <typename RADIAL_BASIS_FUNCTION_T>
class AnisotropicVertexCluster {
public:
    AnisotropicVertexCluster(
        mesh::Vertex center, 
        RADIAL_BASIS_FUNCTION_T function,
        Polynomial polynomial,
        mesh::PtrMesh inputMesh,
        mesh::PtrMesh outputMesh,
        const GlobalAnisotropyParams& params);

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

    double computeD2(const mesh::Vertex &v) const;

    double computeWeight(const mesh::Vertex &v) const;

    Eigen::VectorXd getWeights() const;

    boost::container::flat_set<VertexID> getOutputIDs() const;

    // 判断顶点是否在椭球覆盖范围内，判据: (x-c)^T * M * (x-c) <= 1.0
    bool isCovering(const mesh::Vertex &v) const;

private:
    /// logger, as usual
    mutable precice::logging::Logger _log{"mapping::AnisotropicVertexCluster"};

    /// center vertex of the cluster
    mesh::Vertex _center;

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
    
    // 运行时缓存的逆协方差矩阵，用于快速判断覆盖 (Mahalanobis distance)
    Eigen::Matrix3d _inverseCovariance;
};

// --------------------------------------------------- HEADER IMPLEMENTATIONS

template <typename RADIAL_BASIS_FUNCTION_T>
AnisotropicVertexCluster<RADIAL_BASIS_FUNCTION_T>::AnisotropicVertexCluster(
    mesh::Vertex center, 
    RADIAL_BASIS_FUNCTION_T function,
    Polynomial polynomial,
    mesh::PtrMesh inMesh,
    mesh::PtrMesh outMesh,
    const GlobalAnisotropyParams& params
) : _center(center), _polynomial(polynomial), _function(function), _inverseCovariance(params.inverseCovariance)
{
    // 1. Identify input and output vertex IDs within the anisotropic ellipsoidal region
    // Coarse filter: get vertices inside the bounding box defined by coverSearchRadius
    auto inIDs = inMesh->index().getVerticesInsideBox(center, params.coverSearchRadius);
    auto outIDs = outMesh->index().getVerticesInsideBox(center, params.coverSearchRadius);

    // Fine filter: check isCovering
    auto filterIDs = [&](mesh::Mesh::VertexOffsets& ids, mesh::PtrMesh mesh) {
        for (auto it = ids.begin(); it != ids.end(); ) {
            if (!isCovering(mesh->vertex(*it))) {
                it = ids.erase(it);
            } else {
                ++it;
            }
        }
    };
    
    filterIDs(inIDs, inMesh);
    filterIDs(outIDs, outMesh);

    _inputIDs.insert(inIDs.begin(), inIDs.end());
    _outputIDs.insert(outIDs.begin(), outIDs.end());

    // 2. If the cluster is empty, we return immediately
    if (empty()) {
        return;
    }

    // 3. Re-initialize RBF Solver
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
bool AnisotropicVertexCluster<RADIAL_BASIS_FUNCTION_T>::isCovering(const mesh::Vertex &v) const 
{
    return computeD2(v) <= 1.0;
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
double AnisotropicVertexCluster<RADIAL_BASIS_FUNCTION_T>::computeD2(const mesh::Vertex &v) const
{
    auto c = getCenterCoords();
    Eigen::Vector3d center(c[0], c[1], c[2]);

    Eigen::Vector3d vPos;
    if (v.getDimensions() == 3) {
        vPos = v.getCoords();
    } else {
        vPos << v.coord(0), v.coord(1), 0.0;
    }

    Eigen::Vector3d diff = vPos - center;
    return diff.transpose() * _inverseCovariance * diff;
}

template <typename RADIAL_BASIS_FUNCTION_T>
double AnisotropicVertexCluster<RADIAL_BASIS_FUNCTION_T>::computeWeight(const mesh::Vertex &v) const
{
    const double d2 = computeD2(v);

    if (d2 > 1.0) {
        return 0.0;
    } else {
        // Use CompactPolynomialC2 to ensure monotonic decrease and 0 boundary
        // Formula: (1 - u)^4 * (4u + 1) where u = sqrt(d2)
        // Since we normalized boundary to 1.0, we use supportRadius = 1.0
        static CompactPolynomialC2 weighting(1.0);
        return weighting.evaluate(std::sqrt(d2));
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