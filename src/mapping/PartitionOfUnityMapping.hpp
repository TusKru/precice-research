#pragma once

#include <Eigen/Core>
#include <numeric>

#include "com/Communication.hpp"
#include "io/ExportVTU.hpp"
#include "mapping/impl/AnisotropicClustering.hpp"
#include "mapping/impl/CreateClustering.hpp"
#include "mapping/impl/SphericalVertexCluster.hpp"
#include "mapping/impl/MappingDataCache.hpp"
#include "mapping/impl/AnisotropicVertexCluster.hpp"
#include "mesh/Filter.hpp"
#include "precice/impl/Types.hpp"
#include "profiling/Event.hpp"
#include "query/Index.hpp"
#include "utils/IntraComm.hpp"

namespace precice {
extern bool syncMode;

namespace mapping {

/**
 * Mapping using partition of unity decomposition strategies: The class here inherits from the Mapping
 * class and orchestrates the partitions (called vertex clusters) in order to represent a partition of unity.
 * This means in particular that the class computes the weights for the evaluation vertices and the necessary
 * association between evaluation vertices and the clusters during initialization and traverses through all
 * vertex clusters when evaluating the mapping.
 */
template <typename RADIAL_BASIS_FUNCTION_T>
class PartitionOfUnityMapping : public Mapping {
public:
  /**
   * Constructor, which mostly sets the mesh connectivity requirements and initializes member variables.
   *
   * @param[in] constraint Specifies mapping to be consistent or conservative.
   * @param[in] dimension Dimensionality of the meshes
   * @param[in] function Radial basis function type used in interpolation
   * @param[in] polynomial The handling of the polynomial in the RBF system. Valid choices are 'off' and 'separate'
   * @param[in] verticesPerCluster Target number of vertices to be clustered together
   * @param[in] relativeOverlap Overlap between clusters: The parameter here determines the distance between two cluster
   * centers, given the cluster radius (already determined through \p verticesPerCluster ). A value of 1 would correspond
   * to no distance between cluster centers (i.e. completely overlapping clusters), 0 to distance of 2 x radius between
   * clusters centers.
   * @param[in] projectToInput if enabled, places the cluster centers at the closest vertex of the input mesh.
   * See also \ref mapping::impl::createClustering()
   */
  PartitionOfUnityMapping(
      Mapping::Constraint     constraint,
      int                     dimension,
      RADIAL_BASIS_FUNCTION_T function,
      Polynomial              polynomial,
      unsigned int            verticesPerCluster,
      double                  relativeOverlap,
      bool                    projectToInput,
      bool                    useDynamicRatio = true,
      bool                    useAnisotropic = false,
      bool                    autoFallback = true,
      double                  staticRatio1 = -1.0,
      double                  staticRatio2 = -1.0);

  /**
   * Computes the clustering for the partition of unity method and fills the \p _clusters vector,
   * which allows to travers through all vertex cluster computed. Each vertex cluster in the vector
   * directly computes local mapping matrices and matrix decompositions.
   * In addition, the method computes the normalized weights (Shepard's method) for the partition
   * of unity method and stores them directly in each relevant vertex cluster.
   * In debug mode, the function also exports the partition centers as a separate mesh for visualization
   * purpose.
   */
  void computeMapping() final override;

  /// Clears a computed mapping by deleting the content of the \p _clusters vector.
  void clear() final override;

  /// tag the vertices required for the mapping
  void tagMeshFirstRound() final override;

  /// nothing to do here
  void tagMeshSecondRound() final override;

  /// name of the pum mapping
  std::string getName() const final override;

  void mapConsistentAt(const Eigen::Ref<const Eigen::MatrixXd> &coordinates, const impl::MappingDataCache &cache, Eigen::Ref<Eigen::MatrixXd> values) final override;

  /// the target values here remain unused, as we store the (intermediate) result directly in the cache
  void mapConservativeAt(const Eigen::Ref<const Eigen::MatrixXd> &coordinates, const Eigen::Ref<const Eigen::MatrixXd> &source, impl::MappingDataCache &cache, Eigen::Ref<Eigen::MatrixXd> target) final override;

  void updateMappingDataCache(impl::MappingDataCache &cache, const Eigen::Ref<const Eigen::VectorXd> &in) final override;

  void completeJustInTimeMapping(impl::MappingDataCache &cache, Eigen::Ref<Eigen::MatrixXd> buffer) final override;

  void initializeMappingDataCache(impl::MappingDataCache &cache) final override;

private:
  bool useAnisotropicClusters() const;

  /// logger, as usual
  precice::logging::Logger _log{"mapping::PartitionOfUnityMapping"};

  /// main data container storing all the clusters, which need to be solved individually
  std::vector<AnisotropicVertexCluster<RADIAL_BASIS_FUNCTION_T>> _clusters;
  std::vector<SphericalVertexCluster<RADIAL_BASIS_FUNCTION_T>>   _clustersSpherical;

  /// Radial basis function type used in interpolation
  RADIAL_BASIS_FUNCTION_T _basisFunction;

  /// Input parameters provided by the user for the clustering algorithm:

  /// target number of input vertices for each cluster
  const unsigned int _verticesPerCluster;

  /// overlap of vertex clusters
  const double _relativeOverlap;

  /// toggles whether we project the cluster centers to the input mesh
  const bool _projectToInput;
  const bool _useDynamicRatio;
  const bool _useAnisotropic;
  const bool _autoFallback;
  const double _staticRatio1;
  const double _staticRatio2;

  /// derived parameter based on the input above: the radius of each cluster
  double _coverSearchRadius = 0;
  double _clusterRadius = 0;
  
  GlobalAnisotropyParams _anisoParams;

  /// polynomial treatment of the RBF system
  Polynomial _polynomial;

  std::unique_ptr<mesh::Mesh> _centerMesh;

  /// @copydoc Mapping::mapConservative
  void mapConservative(const time::Sample &inData, Eigen::VectorXd &outData) override;

  /// @copydoc Mapping::mapConsistent
  void mapConsistent(const time::Sample &inData, Eigen::VectorXd &outData) override;

  /// Given an output vertex, computes the normalized PU weight at the given location
  /// The mesh name is only required for logging purposes
  /// returns the clusterIDs and all weights for these clusters
  std::pair<std::vector<int>, std::vector<double>> computeNormalizedWeight(const mesh::Vertex &v, std::string_view mesh);

  void computeNormalizedWeights(mesh::PtrMesh outMesh);

  /// export the center vertices of all clusters as a mesh with some additional data on it such as vertex count
  /// only enabled in debug builds and mainly for debugging purpose
  void exportClusterCentersAsVTU(mesh::Mesh &centers);
};

template <typename RADIAL_BASIS_FUNCTION_T>
PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::PartitionOfUnityMapping(
    Mapping::Constraint     constraint,
    int                     dimension,
    RADIAL_BASIS_FUNCTION_T function,
    Polynomial              polynomial,
    unsigned int            verticesPerCluster,
    double                  relativeOverlap,
    bool                    projectToInput,
    bool                    useDynamicRatio,
    bool                    useAnisotropic,
    bool                    autoFallback,
    double                  staticRatio1,
    double                  staticRatio2)
    : Mapping(constraint, dimension, false, Mapping::InitialGuessRequirement::None),
      _basisFunction(function), _verticesPerCluster(verticesPerCluster), _relativeOverlap(relativeOverlap), _projectToInput(projectToInput), _polynomial(polynomial),
      _useDynamicRatio(useDynamicRatio), _useAnisotropic(useAnisotropic), _autoFallback(autoFallback), _staticRatio1(staticRatio1), _staticRatio2(staticRatio2)
{
  PRECICE_ASSERT(this->getDimensions() <= 3);
  PRECICE_ASSERT(_polynomial != Polynomial::ON, "Integrated polynomial is not supported for partition of unity data mappings.");
  PRECICE_ASSERT(_relativeOverlap < 1, "The relative overlap has to be smaller than one.");
  PRECICE_ASSERT(_verticesPerCluster > 0, "The number of vertices per cluster has to be greater zero.");

  if (isScaledConsistent()) {
    setInputRequirement(Mapping::MeshRequirement::FULL);
    setOutputRequirement(Mapping::MeshRequirement::FULL);
  } else {
    setInputRequirement(Mapping::MeshRequirement::VERTEX);
    setOutputRequirement(Mapping::MeshRequirement::VERTEX);
  }
}

template <typename RADIAL_BASIS_FUNCTION_T>
bool PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::useAnisotropicClusters() const
{
  return _useAnisotropic && !_anisoParams.fallbackToSpherical;
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::computeMapping()
{
  PRECICE_TRACE();
  precice::profiling::Event e("map.pou.computeMapping.From" + this->input()->getName() + "To" + this->output()->getName(), profiling::Synchronize);

  // Recompute the whole clustering
  PRECICE_ASSERT(!this->_hasComputedMapping, "Please clear the mapping before recomputing.");

  mesh::PtrMesh inMesh;
  mesh::PtrMesh outMesh;
  if (this->hasConstraint(Mapping::CONSERVATIVE)) {
    inMesh  = this->output();
    outMesh = this->input();
  } else { // Consistent or scaled consistent
    inMesh  = this->input();
    outMesh = this->output();
  }

  precice::profiling::Event eClusters("map.pou.computeMapping.createClustering.From" + this->input()->getName() + "To" + this->output()->getName());
  std::vector<mesh::Vertex> centerCandidates;
  if (_useAnisotropic) {
    auto [params, centers] = impl::createAnisotropicClustering(inMesh, outMesh, _verticesPerCluster, _relativeOverlap, _projectToInput, _useDynamicRatio, _autoFallback, _staticRatio1, _staticRatio2);
    _anisoParams           = params;
    if (useAnisotropicClusters()) {
      PRECICE_DEBUG("Using anisotropic partition-of-unity clustering with cover search radius {}.", params.coverSearchRadius);
      _coverSearchRadius = params.coverSearchRadius; // Use the second-longest semi-axis for the coarse search radius
      centerCandidates   = std::move(centers);
      PRECICE_ASSERT(_coverSearchRadius > 0 || inMesh->nVertices() == 0 || outMesh->nVertices() == 0);
    } else {
      PRECICE_INFO("Anisotropic clustering requested, but the global anisotropy analysis triggered spherical fallback. Recomputing spherical clusters.");
      auto [radius, sphericalCenters] = impl::createClustering(inMesh, outMesh, _relativeOverlap, _verticesPerCluster, _projectToInput);
      _clusterRadius                  = radius;
      _coverSearchRadius              = radius;
      centerCandidates                = std::move(sphericalCenters);
      PRECICE_ASSERT(_clusterRadius > 0 || inMesh->nVertices() == 0 || outMesh->nVertices() == 0);
    }
  } else {
    auto [radius, centers] = impl::createClustering(inMesh, outMesh, _relativeOverlap, _verticesPerCluster, _projectToInput);
    _clusterRadius         = radius;
    centerCandidates       = std::move(centers);
    PRECICE_ASSERT(_clusterRadius > 0 || inMesh->nVertices() == 0 || outMesh->nVertices() == 0);
  }
  eClusters.stop();

  // Step 2: check, which of the resulting clusters are non-empty and register the cluster centers in a mesh
  // Here, the VertexCluster computes the matrix decompositions directly in case the cluster is non-empty
  _centerMesh        = std::make_unique<mesh::Mesh>("pou-centers-" + inMesh->getName(), this->getDimensions(), mesh::Mesh::MESH_ID_UNDEFINED);
  auto &meshVertices = _centerMesh->vertices();
  
  meshVertices.clear();
  _clusters.clear();
  _clustersSpherical.clear();
  if (useAnisotropicClusters()) {
    _clusters.reserve(centerCandidates.size());
    for (const auto &c : centerCandidates) {
      const VertexID                                  vertexID = meshVertices.size();
      mesh::Vertex                                    center(c.getCoords(), vertexID);
      AnisotropicVertexCluster<RADIAL_BASIS_FUNCTION_T> cluster(center, _basisFunction, _polynomial, inMesh, outMesh, _anisoParams);
      if (!cluster.empty()) {
        PRECICE_ASSERT(center.getID() == static_cast<int>(_clusters.size()), center.getID(), _clusters.size());
        meshVertices.emplace_back(std::move(center));
        _clusters.emplace_back(std::move(cluster));
      }
    }
  } else {
    _clustersSpherical.reserve(centerCandidates.size());
    for (const auto &c : centerCandidates) {
      const VertexID                                vertexID = meshVertices.size();
      mesh::Vertex                                  center(c.getCoords(), vertexID);
      SphericalVertexCluster<RADIAL_BASIS_FUNCTION_T> cluster(center, _clusterRadius, _basisFunction, _polynomial, inMesh, outMesh);
      if (!cluster.empty()) {
        PRECICE_ASSERT(center.getID() == static_cast<int>(_clustersSpherical.size()), center.getID(), _clustersSpherical.size());
        meshVertices.emplace_back(std::move(center));
        _clustersSpherical.emplace_back(std::move(cluster));
      }
    }
  }

  e.addData("n clusters", useAnisotropicClusters() ? _clusters.size() : _clustersSpherical.size());
  // Log the average number of resulting clusters
  
  if (useAnisotropicClusters()) {
    if (_clusters.size() > 0) {
      PRECICE_DEBUG("Partition of unity data mapping between mesh \"{}\" and mesh \"{}\": mesh \"{}\" on rank {} was decomposed into {} clusters.", this->input()->getName(), this->output()->getName(), inMesh->getName(), utils::IntraComm::getRank(), _clusters.size());

      PRECICE_DEBUG("Average number of vertices per cluster {}", std::accumulate(_clusters.begin(), _clusters.end(), static_cast<unsigned int>(0), [](auto &acc, auto &val) { return acc += val.getNumberOfInputVertices(); }) / _clusters.size());
      PRECICE_DEBUG("Maximum number of vertices per cluster {}", std::max_element(_clusters.begin(), _clusters.end(), [](auto &v1, auto &v2) { return v1.getNumberOfInputVertices() < v2.getNumberOfInputVertices(); })->getNumberOfInputVertices());
      PRECICE_DEBUG("Minimum number of vertices per cluster {}", std::min_element(_clusters.begin(), _clusters.end(), [](auto &v1, auto &v2) { return v1.getNumberOfInputVertices() < v2.getNumberOfInputVertices(); })->getNumberOfInputVertices());
    }
  } else {
    if (_clustersSpherical.size() > 0) {
      PRECICE_DEBUG("Partition of unity data mapping between mesh \"{}\" and mesh \"{}\": mesh \"{}\" on rank {} was decomposed into {} clusters.", this->input()->getName(), this->output()->getName(), inMesh->getName(), utils::IntraComm::getRank(), _clustersSpherical.size());

      PRECICE_DEBUG("Average number of vertices per cluster {}", std::accumulate(_clustersSpherical.begin(), _clustersSpherical.end(), static_cast<unsigned int>(0), [](auto &acc, auto &val) { return acc += val.getNumberOfInputVertices(); }) / _clustersSpherical.size());
      PRECICE_DEBUG("Maximum number of vertices per cluster {}", std::max_element(_clustersSpherical.begin(), _clustersSpherical.end(), [](auto &v1, auto &v2) { return v1.getNumberOfInputVertices() < v2.getNumberOfInputVertices(); })->getNumberOfInputVertices());
      PRECICE_DEBUG("Minimum number of vertices per cluster {}", std::min_element(_clustersSpherical.begin(), _clustersSpherical.end(), [](auto &v1, auto &v2) { return v1.getNumberOfInputVertices() < v2.getNumberOfInputVertices(); })->getNumberOfInputVertices());
    }
  }

  precice::profiling::Event eWeights("map.pou.computeMapping.computeWeights");
  // Log a bounding box of the center mesh
  _centerMesh->computeBoundingBox();
  PRECICE_DEBUG("Bounding Box of the cluster centers {}", _centerMesh->getBoundingBox());

  // Step 3: Determine PU weights
  PRECICE_DEBUG("Computing cluster-vertex association");
  for (const auto &vertex : outMesh->vertices()) {
    auto [clusterIDs, normalizedWeights] = computeNormalizedWeight(vertex, outMesh->getName());
    for (unsigned int i = 0; i < clusterIDs.size(); ++i) {
      if (useAnisotropicClusters()) {
        PRECICE_ASSERT(clusterIDs[i] < static_cast<int>(_clusters.size()));
        _clusters[clusterIDs[i]].setNormalizedWeight(normalizedWeights[i], vertex.getID());
      } else {
        PRECICE_ASSERT(clusterIDs[i] < static_cast<int>(_clustersSpherical.size()));
        _clustersSpherical[clusterIDs[i]].setNormalizedWeight(normalizedWeights[i], vertex.getID());
      }
    }
  }
  eWeights.stop();

  // Uncomment to add a VTK export of the cluster center distribution for visualization purposes
  exportClusterCentersAsVTU(*_centerMesh);

  // we need the center mesh index data structure
  if (!outMesh->isJustInTime()) {
    _centerMesh.reset();
  }

  this->_hasComputedMapping = true;
}

template <typename RADIAL_BASIS_FUNCTION_T>
std::pair<std::vector<int>, std::vector<double>> PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::computeNormalizedWeight(const mesh::Vertex &vertex, std::string_view mesh)
{

  // Step 1: index the clusters / the center mesh in order to define the output vertex -> cluster ownership
  // the ownership is required to compute the normalized partition of unity weights (Step 2)
  // query::Index clusterIndex(*_centerMesh.get());
  PRECICE_ASSERT(_centerMesh);
  query::Index &clusterIndex = _centerMesh->index();

  // Step 2: find all clusters the output vertex lies in
  // Step 2a: get the relevant clusters for the output vertex (COARSE FILTER)
  auto       candidateClusterIDs   = clusterIndex.getVerticesInsideBox(vertex, useAnisotropicClusters() ? _coverSearchRadius : _clusterRadius);
  // PRECICE_INFO("candidateClusterIDs.size() = {}", candidateClusterIDs.size());

  // Step 2b: Fine Filter + Compute Weights
  std::vector<int> clusterIDs;
  std::vector<double> weights;
  
  for(auto id : candidateClusterIDs) {
    if (useAnisotropicClusters()) {
      if(_clusters[id].isCovering(vertex)) {
          clusterIDs.push_back(id);
          weights.push_back(_clusters[id].computeWeight(vertex));
      }
    } else {
      clusterIDs.push_back(id);
      weights.push_back(_clustersSpherical[id].computeWeight(vertex));
    }
  }
  
  const auto localNumberOfClusters = clusterIDs.size();
  // PRECICE_INFO("Output vertex {} of mesh \"{}\": localNumberOfClusters = {}", vertex.getCoords(), mesh, localNumberOfClusters);

  // Consider the case where we didn't find any cluster (meshes don't match very well)
  PRECICE_CHECK(localNumberOfClusters > 0,
                "Output vertex {} of mesh \"{}\" could not be assigned to any cluster in the rbf-pum mapping. This probably means that the meshes do not match well geometry-wise: Visualize the exported preCICE meshes to confirm. "
                "If the meshes are fine geometry-wise, you can try to increase the number of \"vertices-per-cluster\" (default is 50), the \"relative-overlap\" (default is 0.15), or disable the option \"project-to-input\". "
                "These options are only valid for the <mapping:rbf-pum-direct/> tag.",
                vertex.getCoords(), mesh);

  // Next we compute the normalized weights of each output vertex for each partition
  PRECICE_ASSERT(localNumberOfClusters > 0, "No cluster found for vertex {}", vertex.getCoords());

  double weightSum = std::accumulate(weights.begin(), weights.end(), static_cast<double>(0.));
  // TODO: This covers the edge case of vertices being at the edge of (several) clusters
  // In case the sum is equal to zero, we assign equal weights for all clusters
  if (weightSum <= 0) {
    PRECICE_ASSERT(weights.size() > 0);
    std::for_each(weights.begin(), weights.end(), [&weights](auto &w) { w = 1. / weights.size(); });
    weightSum = 1;
  }
  PRECICE_ASSERT(weightSum > 0);

  // Step 2c: Normalize weights
  std::transform(weights.begin(), weights.end(), weights.begin(), [weightSum](double w) { return w / weightSum; });

  // Return both the cluster IDs and the normalized weights
  return {clusterIDs, weights};
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::mapConservative(const time::Sample &inData, Eigen::VectorXd &outData)
{
  PRECICE_TRACE();

  precice::profiling::Event e("map.pou.mapData.From" + input()->getName() + "To" + output()->getName(), profiling::Synchronize);

  // Execute the actual mapping evaluation in all clusters
  // 1. Assert that all output data values were reset, as we accumulate data in all clusters independently
  PRECICE_ASSERT(outData.isZero());

  // 2. Iterate over all clusters and accumulate the result in the output data
  if (useAnisotropicClusters()) {
    std::for_each(_clusters.begin(), _clusters.end(), [&](auto &cluster) { cluster.mapConservative(inData, outData); });
  } else {
    std::for_each(_clustersSpherical.begin(), _clustersSpherical.end(), [&](auto &cluster) { cluster.mapConservative(inData, outData); });
  }
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::mapConsistent(const time::Sample &inData, Eigen::VectorXd &outData)
{
  PRECICE_TRACE();

  precice::profiling::Event e("map.pou.mapData.From" + input()->getName() + "To" + output()->getName(), profiling::Synchronize);

  // Execute the actual mapping evaluation in all clusters
  // 1. Assert that all output data values were reset, as we accumulate data in all clusters independently
  PRECICE_ASSERT(outData.isZero());

  // 2. Execute the actual mapping evaluation in all vertex clusters and accumulate the data
  if (useAnisotropicClusters()) {
    std::for_each(_clusters.begin(), _clusters.end(), [&](auto &clusters) { clusters.mapConsistent(inData, outData); });
  } else {
    std::for_each(_clustersSpherical.begin(), _clustersSpherical.end(), [&](auto &clusters) { clusters.mapConsistent(inData, outData); });
  }
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::mapConservativeAt(const Eigen::Ref<const Eigen::MatrixXd> &coordinates, const Eigen::Ref<const Eigen::MatrixXd> &source,
                                                                         impl::MappingDataCache &cache, Eigen::Ref<Eigen::MatrixXd>)
{
  precice::profiling::Event e("map.pou.mapConservativeAt.From" + input()->getName());
  // @todo: it would most probably be more efficient to first group the vertices we receive here according to the clusters and then compute the solution

  PRECICE_TRACE();
  PRECICE_ASSERT(_centerMesh);
  PRECICE_ASSERT(cache.p.size() == (useAnisotropicClusters() ? _clusters.size() : _clustersSpherical.size()));
  PRECICE_ASSERT(cache.polynomialContributions.size() == (useAnisotropicClusters() ? _clusters.size() : _clustersSpherical.size()));

  mesh::Vertex vertex(coordinates.col(0), -1);
  for (Eigen::Index v = 0; v < coordinates.cols(); ++v) {
    vertex.setCoords(coordinates.col(v));
    auto [clusterIDs, normalizedWeights] = computeNormalizedWeight(vertex, this->input()->getName());
    // Use the weight to interpolate the solution
    for (std::size_t i = 0; i < clusterIDs.size(); ++i) {
      auto id = clusterIDs[i];
      Eigen::VectorXd res = normalizedWeights[i] * source.col(v);
      if (useAnisotropicClusters()) {
        PRECICE_ASSERT(id < static_cast<int>(_clusters.size()));
        _clusters[id].addWriteDataToCache(vertex, res, cache.polynomialContributions[id], cache.p[id], *this->output().get());
      } else {
        PRECICE_ASSERT(id < static_cast<int>(_clustersSpherical.size()));
        _clustersSpherical[id].addWriteDataToCache(vertex, res, cache.polynomialContributions[id], cache.p[id], *this->output().get());
      }
    }
  }
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::completeJustInTimeMapping(impl::MappingDataCache &cache, Eigen::Ref<Eigen::MatrixXd> buffer)
{
  PRECICE_TRACE();
  PRECICE_ASSERT(!cache.p.empty());
  PRECICE_ASSERT(!cache.polynomialContributions.empty());
  precice::profiling::Event e("map.pou.completeJustInTimeMapping.From" + input()->getName());

  if (useAnisotropicClusters()) {
    for (std::size_t c = 0; c < _clusters.size(); ++c) {
      if (cache.p[c].squaredNorm() > 0) {
        _clusters[c].evaluateConservativeCache(cache.polynomialContributions[c], cache.p[c], buffer);
      }
    }
  } else {
    for (std::size_t c = 0; c < _clustersSpherical.size(); ++c) {
      if (cache.p[c].squaredNorm() > 0) {
        _clustersSpherical[c].evaluateConservativeCache(cache.polynomialContributions[c], cache.p[c], buffer);
      }
    }
  }
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::initializeMappingDataCache(impl::MappingDataCache &cache)
{
  PRECICE_TRACE();
  PRECICE_ASSERT(_hasComputedMapping);
  const auto nC = useAnisotropicClusters() ? _clusters.size() : _clustersSpherical.size();
  cache.p.resize(nC);
  cache.polynomialContributions.resize(nC);
  if (useAnisotropicClusters()) {
    for (std::size_t c = 0; c < _clusters.size(); ++c) {
      _clusters[c].initializeCacheData(cache.polynomialContributions[c], cache.p[c], cache.getDataDimensions());
    }
  } else {
    for (std::size_t c = 0; c < _clustersSpherical.size(); ++c) {
      _clustersSpherical[c].initializeCacheData(cache.polynomialContributions[c], cache.p[c], cache.getDataDimensions());
    }
  }
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::updateMappingDataCache(impl::MappingDataCache &cache, const Eigen::Ref<const Eigen::VectorXd> &in)
{
  // We cannot synchronize this event, as the call to this function is rank-local only
  precice::profiling::Event e("map.pou.updateMappingDataCache.From" + input()->getName());
  const auto nC = useAnisotropicClusters() ? _clusters.size() : _clustersSpherical.size();
  PRECICE_ASSERT(cache.p.size() == nC);
  PRECICE_ASSERT(cache.polynomialContributions.size() == nC);
  Eigen::Map<const Eigen::MatrixXd> inMatrix(in.data(), cache.getDataDimensions(), in.size() / cache.getDataDimensions());
  if (useAnisotropicClusters()) {
    for (std::size_t c = 0; c < _clusters.size(); ++c) {
      _clusters[c].computeCacheData(inMatrix, cache.polynomialContributions[c], cache.p[c]);
    }
  } else {
    for (std::size_t c = 0; c < _clustersSpherical.size(); ++c) {
      _clustersSpherical[c].computeCacheData(inMatrix, cache.polynomialContributions[c], cache.p[c]);
    }
  }
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::mapConsistentAt(const Eigen::Ref<const Eigen::MatrixXd> &coordinates, const impl::MappingDataCache &cache, Eigen::Ref<Eigen::MatrixXd> values)
{
  precice::profiling::Event e("map.pou.mapConsistentAt.From" + input()->getName());
  // @todo: it would most probably be more efficient to first group the vertices we receive here according to the clusters and then compute the solution
  PRECICE_TRACE();
  PRECICE_ASSERT(_centerMesh);

  // First, make sure that everything is reset before we start
  values.setZero();

  mesh::Vertex vertex(coordinates.col(0), -1);
  for (Eigen::Index v = 0; v < values.cols(); ++v) {
    vertex.setCoords(coordinates.col(v));
    auto [clusterIDs, normalizedWeights] = computeNormalizedWeight(vertex, this->output()->getName());
    // Use the weight to interpolate the solution
    for (std::size_t i = 0; i < clusterIDs.size(); ++i) {
      auto id = clusterIDs[i];
      Eigen::VectorXd localRes;
      if (useAnisotropicClusters()) {
        PRECICE_ASSERT(clusterIDs[i] < static_cast<int>(_clusters.size()));
        localRes = normalizedWeights[i] * _clusters[id].interpolateAt(vertex, cache.polynomialContributions[id], cache.p[id], *this->input().get());
      } else {
        PRECICE_ASSERT(clusterIDs[i] < static_cast<int>(_clustersSpherical.size()));
        localRes = normalizedWeights[i] * _clustersSpherical[id].interpolateAt(vertex, cache.polynomialContributions[id], cache.p[id], *this->input().get());
      }
      values.col(v) += localRes;
    }
  }
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::tagMeshFirstRound()
{
  PRECICE_TRACE();
  mesh::PtrMesh filterMesh, outMesh;
  if (this->hasConstraint(Mapping::CONSERVATIVE)) {
    filterMesh = this->output(); // remote
    outMesh    = this->input();  // local
  } else {
    filterMesh = this->input();  // remote
    outMesh    = this->output(); // local
  }

  if (outMesh->empty())
    return; // Ranks not at the interface should never hold interface vertices

  // The geometric filter of the repartitioning is always disabled for the PU-RBF.
  // The main rationale: if we use only a fraction of the mesh then we might end up
  // with too few vertices per rank and we cannot prevent too few vertices from being
  // tagged, if we have filtered too much vertices beforehand. When using the filtering,
  // the user could increase the safety-factor or disable the filtering, but that's
  // a bit hard to understand for users. When no geometric filter is applid,
  // vertices().size() is here the same as getGlobalNumberOfVertices. Hence, it is much
  // safer to make use of the unfiltered mesh for the parallel tagging.
  //
  // Drawback: the "estimateClusterRadius" below makes use of the mesh R* index tree, and
  // constructing the tree on the (unfiltered) global mesh is computationally expensive (O( N logN)).
  // We could pre-filter the global mesh to a local fraction (using our own geometric filtering,
  // maybe with an increased safety margin or even an iterative increase of the safety margin),
  // but then there is again the question on how to do this in a safe way, without risking
  // failures depending on the partitioning. So we stick here to the computationally more
  // demanding, but safer version.
  // See also https://github.com/precice/precice/pull/1912#issuecomment-2551143620

  // Get the local bounding boxes
  auto localBB = outMesh->getBoundingBox();
  // we cannot check for empty'ness here, as a single output mesh vertex
  // would lead to a 0D box with zero volume (considered empty). Thus, we
  // simply check here for default'ness, which is equivalent to outMesh->empty()
  // further above
  PRECICE_ASSERT(!localBB.isDefault());

  if (_coverSearchRadius == 0)
    _coverSearchRadius = impl::estimateClusterRadius(_verticesPerCluster, filterMesh, localBB); // Use the new estimateBaseRadius

  PRECICE_DEBUG("Cluster radius estimate: {}", _coverSearchRadius);
  PRECICE_ASSERT(_coverSearchRadius > 0);

  // Now we extend the bounding box by the radius
  localBB.expandBy(2 * _coverSearchRadius);

  // ... and tag all affected vertices
  auto verticesNew = filterMesh->index().getVerticesInsideBox(localBB);

  std::for_each(verticesNew.begin(), verticesNew.end(), [&filterMesh](VertexID v) { filterMesh->vertex(v).tag(); });
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::tagMeshSecondRound()
{
  // Nothing to be done here. There is no global ownership for matrix entries required and we tag all potentially locally relevant vertices already in the first round.
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::exportClusterCentersAsVTU(mesh::Mesh &centerMesh)
{
  PRECICE_TRACE();

  auto dataRadius      = centerMesh.createData("radius", 1, -1);
  auto dataCardinality = centerMesh.createData("number-of-vertices", 1, -1);
  centerMesh.allocateDataValues();
  dataRadius->values().fill(useAnisotropicClusters() ? _coverSearchRadius : _clusterRadius);
  if (useAnisotropicClusters()) {
    for (unsigned int i = 0; i < _clusters.size(); ++i) {
      dataCardinality->values()[i] = static_cast<double>(_clusters[i].getNumberOfInputVertices());
    }
  } else {
    for (unsigned int i = 0; i < _clustersSpherical.size(); ++i) {
      dataCardinality->values()[i] = static_cast<double>(_clustersSpherical[i].getNumberOfInputVertices());
    }
  }

  // We have to create the global offsets in order to export things in parallel
  if (utils::IntraComm::isSecondary()) {
    // send number of vertices
    PRECICE_DEBUG("Send number of vertices: {}", centerMesh.nVertices());
    int numberOfVertices = centerMesh.nVertices();
    utils::IntraComm::getCommunication()->send(numberOfVertices, 0);

    // receive vertex offsets
    mesh::Mesh::VertexOffsets vertexOffsets;
    utils::IntraComm::getCommunication()->broadcast(vertexOffsets, 0);
    PRECICE_DEBUG("Vertex offsets: {}", vertexOffsets);
    PRECICE_ASSERT(centerMesh.getVertexOffsets().empty());
    centerMesh.setVertexOffsets(std::move(vertexOffsets));
  } else if (utils::IntraComm::isPrimary()) {

    mesh::Mesh::VertexOffsets vertexOffsets(utils::IntraComm::getSize());
    vertexOffsets[0] = centerMesh.nVertices();

    // receive number of secondary vertices and fill vertex offsets
    for (int secondaryRank : utils::IntraComm::allSecondaryRanks()) {
      int numberOfSecondaryRankVertices = -1;
      utils::IntraComm::getCommunication()->receive(numberOfSecondaryRankVertices, secondaryRank);
      PRECICE_ASSERT(numberOfSecondaryRankVertices >= 0);
      vertexOffsets[secondaryRank] = numberOfSecondaryRankVertices + vertexOffsets[secondaryRank - 1];
    }

    // broadcast vertex offsets
    PRECICE_DEBUG("Vertex offsets: {}", centerMesh.getVertexOffsets());
    utils::IntraComm::getCommunication()->broadcast(vertexOffsets);
    centerMesh.setVertexOffsets(std::move(vertexOffsets));
  }

  dataRadius->setSampleAtTime(0, time::Sample{1, dataRadius->values()});
  dataCardinality->setSampleAtTime(0, time::Sample{1, dataCardinality->values()});
  io::ExportVTU exporter{"PoU", "exports", centerMesh, io::Export::ExportKind::TimeWindows, 1, utils::IntraComm::getRank(), utils::IntraComm::getSize()};
  exporter.doExport(0, 0.0);
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::clear()
{
  PRECICE_TRACE();
  _clusters.clear();
  _clustersSpherical.clear();
  // TODO: Don't reset this here
  _coverSearchRadius            = 0;
  _clusterRadius                = 0;
  this->_hasComputedMapping = false;
}

template <typename RADIAL_BASIS_FUNCTION_T>
std::string PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::getName() const
{
  return "partition-of-unity RBF";
}
} // namespace mapping
} // namespace precice
