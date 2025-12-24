#pragma once

#include <Eigen/Core>
#include <numeric>

#include "com/Communication.hpp"
#include "io/ExportVTU.hpp"
#include "mapping/impl/CreateClustering.hpp"
#include "mapping/impl/MappingDataCache.hpp"
#include "mapping/impl/SphericalVertexCluster.hpp"
#include "mesh/Filter.hpp"
#include "precice/impl/Types.hpp"
#include "profiling/Event.hpp"
#include "query/Index.hpp"
#include "utils/IntraComm.hpp"

#include "mapping/impl/AnisotropicVertexCluster.hpp"
#include "mapping/impl/AnisotropicClustering.hpp"

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
      bool                    projectToInput);

  /**
   * Computes the clustering for the partition of unity method and fills the \p _anisotropicClusters vector,
   * which allows to travers through all vertex cluster computed. Each vertex cluster in the vector
   * directly computes local mapping matrices and matrix decompositions.
   * In addition, the method computes the normalized weights (Shepard's method) for the partition
   * of unity method and stores them directly in each relevant vertex cluster.
   * In debug mode, the function also exports the partition centers as a separate mesh for visualization
   * purpose.
   */
  void computeMapping() final override;

  /// Clears a computed mapping by deleting the content of the \p _anisotropicClusters vector.
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
  /// logger, as usual
  precice::logging::Logger _log{"mapping::PartitionOfUnityMapping"};

  /// main data container storing all the clusters, which need to be solved individually
  std::vector<AnisotropicVertexCluster<RADIAL_BASIS_FUNCTION_T>> _anisotropicClusters;

  /// Radial basis function type used in interpolation
  RADIAL_BASIS_FUNCTION_T _basisFunction;

  /// Input parameters provided by the user for the clustering algorithm:

  /// target number of input vertices for each cluster
  const unsigned int _verticesPerCluster;

  /// overlap of vertex clusters
  const double _relativeOverlap;

  /// toggles whether we project the cluster centers to the input mesh
  const bool _projectToInput;

  /// derived parameter based on the input above: the radius of each cluster
  double _clusterRadius = 0;

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
    bool                    projectToInput)
    : Mapping(constraint, dimension, false, Mapping::InitialGuessRequirement::None),
      _basisFunction(function), _verticesPerCluster(verticesPerCluster), _relativeOverlap(relativeOverlap), _projectToInput(projectToInput), _polynomial(polynomial)
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

  precice::profiling::Event eClusters("map.pou.computeMapping.createAnisotropicClustering.From" + this->input()->getName() + "To" + this->output()->getName());
  // Step 1: get a tentative clustering consisting of centers and a radius from one of the available algorithms
  _anisotropicClusters = std::move(createAnisotropicClustering<RADIAL_BASIS_FUNCTION_T>(inMesh, outMesh, _basisFunction, _polynomial, _verticesPerCluster));
  eClusters.stop();

  e.addData("n clusters", _anisotropicClusters.size());
  // Log the average number of resulting clusters
  PRECICE_DEBUG("Partition of unity data mapping between mesh \"{}\" and mesh \"{}\": mesh \"{}\" on rank {} was decomposed into {} clusters.", this->input()->getName(), this->output()->getName(), inMesh->getName(), utils::IntraComm::getRank(), _anisotropicClusters.size());

  if (_anisotropicClusters.size() > 0) {
    PRECICE_DEBUG("Average number of vertices per cluster {}", std::accumulate(_anisotropicClusters.begin(), _anisotropicClusters.end(), static_cast<unsigned int>(0), [](auto &acc, auto &val) { return acc += val.getNumberOfInputVertices(); }) / _anisotropicClusters.size());
    PRECICE_DEBUG("Maximum number of vertices per cluster {}", std::max_element(_anisotropicClusters.begin(), _anisotropicClusters.end(), [](auto &v1, auto &v2) { return v1.getNumberOfInputVertices() < v2.getNumberOfInputVertices(); })->getNumberOfInputVertices());
    PRECICE_DEBUG("Minimum number of vertices per cluster {}", std::min_element(_anisotropicClusters.begin(), _anisotropicClusters.end(), [](auto &v1, auto &v2) { return v1.getNumberOfInputVertices() < v2.getNumberOfInputVertices(); })->getNumberOfInputVertices());
  }

  // Step 2: register the cluster centers in a mesh
  // Here, the VertexCluster computes the matrix decompositions directly in case the cluster is non-empty
  _centerMesh        = std::make_unique<mesh::Mesh>("pou-centers-" + inMesh->getName(), this->getDimensions(), mesh::Mesh::MESH_ID_UNDEFINED);
  auto &meshVertices = _centerMesh->vertices();

  meshVertices.clear();

  for (const auto &c : _anisotropicClusters) {
    // We cannot simply copy the vertex from the container in order to fill the vertices of the centerMesh, as the vertexID of each center needs to match the index
    // of the cluster within the _anisotropicClusters vector. That's required for the indexing further down and asserted below
    const VertexID                                  vertexID = meshVertices.size();
    mesh::Vertex                                    center(c.getCenterCoords(), vertexID);

    PRECICE_ASSERT(center.getID() == static_cast<int>(_anisotropicClusters.size()), center.getID(), _anisotropicClusters.size());
    meshVertices.emplace_back(std::move(center));
  }

  precice::profiling::Event eWeights("map.pou.computeMapping.computeWeights");
  computeNormalizedWeights(outMesh);
  eWeights.stop();

  // Uncomment to add a VTK export of the cluster center distribution for visualization purposes
  // exportClusterCentersAsVTU(*_centerMesh);

  // we need the center mesh index data structure
  if (!outMesh->isJustInTime()) {
    _centerMesh.reset();
  }

  this->_hasComputedMapping = true;
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::computeNormalizedWeights(mesh::PtrMesh outMesh)
{
  PRECICE_TRACE();

  // 1-遍历所有簇，提取权重并加到全局权重表中
  std::vector<double> globalWeights(outMesh->nVertices(), 0.0);
  for (const auto &cluster : _anisotropicClusters) {
    const auto &localWeights = cluster.getWeights();
    const auto &outputIDs    = cluster.getOutputIDs();
    for (unsigned int i = 0; i < outputIDs.size(); ++i) {
      const double w = localWeights[i];
      globalWeights[*(outputIDs.nth(i))] += w;
    }
  }

  // 2-遍历所有簇，将权重除以全局权重表中对应项，得到归一化权重并存储回簇中
  for (auto &cluster : _anisotropicClusters) {
    const auto &localWeights = cluster.getWeights();
    const auto &outputIDs    = cluster.getOutputIDs();
    for (unsigned int i = 0; i < outputIDs.size(); ++i) {
      const double normalizedWeight = localWeights[i] / globalWeights[*(outputIDs.nth(i))];
      cluster.setNormalizedWeight(normalizedWeight, *(outputIDs.nth(i)));
    }
  }
}

template <typename RADIAL_BASIS_FUNCTION_T>
std::pair<std::vector<int>, std::vector<double>> PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::computeNormalizedWeight(const mesh::Vertex &vertex, std::string_view mesh)
{
  std::vector<VertexID> clusterIDs;

  for (unsigned int i = 0; i < _anisotropicClusters.size(); ++i) {
    if (_anisotropicClusters[i].isCovering(vertex.getCoords()))
      clusterIDs.push_back(i);
  }
  const auto localNumberOfClusters = clusterIDs.size();

  // Consider the case where we didn't find any cluster (meshes don't match very well)
  //
  // In principle, we could assign the vertex to the closest cluster using clusterIDs.emplace_back(clusterIndex.getClosestVertex(vertex.getCoords()).index);
  // However, this leads to a conflict with weights already set in the corresponding cluster, since we insert the ID and, later on, map the ID to a local weight index
  // Of course, we could rearrange the weights, but we want to avoid the case here anyway, i.e., prefer to abort.
  PRECICE_CHECK(localNumberOfClusters > 0,
                "Output vertex {} of mesh \"{}\" could not be assigned to any cluster in the rbf-pum mapping. This probably means that the meshes do not match well geometry-wise: Visualize the exported preCICE meshes to confirm. "
                "If the meshes are fine geometry-wise, you can try to increase the number of \"vertices-per-cluster\" (default is 50), the \"relative-overlap\" (default is 0.15), or disable the option \"project-to-input\". "
                "These options are only valid for the <mapping:rbf-pum-direct/> tag.",
                vertex.getCoords(), mesh);

  // Next we compute the normalized weights of each output vertex for each partition
  PRECICE_ASSERT(localNumberOfClusters > 0, "No cluster found for vertex {}", vertex.getCoords());

  // Step 2b: compute the weight in each partition individually and store them in 'weights'
  std::vector<double> weights(localNumberOfClusters);
  std::transform(clusterIDs.cbegin(), clusterIDs.cend(), weights.begin(), [&](const auto &ids) { return _anisotropicClusters[ids].computeWeight(vertex); });
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
  std::for_each(_anisotropicClusters.begin(), _anisotropicClusters.end(), [&](auto &cluster) { cluster.mapConservative(inData, outData); });
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
  std::for_each(_anisotropicClusters.begin(), _anisotropicClusters.end(), [&](auto &clusters) { clusters.mapConsistent(inData, outData); });
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::mapConservativeAt(const Eigen::Ref<const Eigen::MatrixXd> &coordinates, const Eigen::Ref<const Eigen::MatrixXd> &source,
                                                                         impl::MappingDataCache &cache, Eigen::Ref<Eigen::MatrixXd>)
{
  precice::profiling::Event e("map.pou.mapConservativeAt.From" + input()->getName());
  // @todo: it would most probably be more efficient to first group the vertices we receive here according to the clusters and then compute the solution

  PRECICE_TRACE();
  PRECICE_ASSERT(_centerMesh);
  PRECICE_ASSERT(cache.p.size() == _anisotropicClusters.size());
  PRECICE_ASSERT(cache.polynomialContributions.size() == _anisotropicClusters.size());

  mesh::Vertex vertex(coordinates.col(0), -1);
  for (Eigen::Index v = 0; v < coordinates.cols(); ++v) {
    vertex.setCoords(coordinates.col(v));
    auto [clusterIDs, normalizedWeights] = computeNormalizedWeight(vertex, this->input()->getName());
    // Use the weight to interpolate the solution
    for (std::size_t i = 0; i < clusterIDs.size(); ++i) {
      PRECICE_ASSERT(clusterIDs[i] < static_cast<int>(_anisotropicClusters.size()));
      auto id = clusterIDs[i];
      // the input mesh refers here to a consistent constraint
      Eigen::VectorXd res = normalizedWeights[i] * source.col(v);
      _anisotropicClusters[id].addWriteDataToCache(vertex, res, cache.polynomialContributions[id], cache.p[id], *this->output().get());
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

  for (std::size_t c = 0; c < _anisotropicClusters.size(); ++c) {
    // If there is no contribution, we don't have to evaluate
    if (cache.p[c].squaredNorm() > 0) {
      _anisotropicClusters[c].evaluateConservativeCache(cache.polynomialContributions[c], cache.p[c], buffer);
    }
  }
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::initializeMappingDataCache(impl::MappingDataCache &cache)
{
  PRECICE_TRACE();
  PRECICE_ASSERT(_hasComputedMapping);
  cache.p.resize(_anisotropicClusters.size());
  cache.polynomialContributions.resize(_anisotropicClusters.size());
  for (std::size_t c = 0; c < _anisotropicClusters.size(); ++c) {
    _anisotropicClusters[c].initializeCacheData(cache.polynomialContributions[c], cache.p[c], cache.getDataDimensions());
  }
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::updateMappingDataCache(impl::MappingDataCache &cache, const Eigen::Ref<const Eigen::VectorXd> &in)
{
  // We cannot synchronize this event, as the call to this function is rank-local only
  precice::profiling::Event e("map.pou.updateMappingDataCache.From" + input()->getName());
  PRECICE_ASSERT(cache.p.size() == _anisotropicClusters.size());
  PRECICE_ASSERT(cache.polynomialContributions.size() == _anisotropicClusters.size());
  Eigen::Map<const Eigen::MatrixXd> inMatrix(in.data(), cache.getDataDimensions(), in.size() / cache.getDataDimensions());
  for (std::size_t c = 0; c < _anisotropicClusters.size(); ++c) {
    _anisotropicClusters[c].computeCacheData(inMatrix, cache.polynomialContributions[c], cache.p[c]);
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
      PRECICE_ASSERT(clusterIDs[i] < static_cast<int>(_anisotropicClusters.size()));
      auto id = clusterIDs[i];
      // the input mesh refers here to a consistent constraint
      Eigen::VectorXd localRes = normalizedWeights[i] * _anisotropicClusters[id].interpolateAt(vertex, cache.polynomialContributions[id], cache.p[id], *this->input().get());
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

  if (_clusterRadius == 0)
    _clusterRadius = impl::estimateClusterRadius(_verticesPerCluster, filterMesh, localBB);

  PRECICE_DEBUG("Cluster radius estimate: {}", _clusterRadius);
  PRECICE_ASSERT(_clusterRadius > 0);

  // Now we extend the bounding box by the radius
  localBB.expandBy(2 * _clusterRadius);

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
  dataRadius->values().fill(_clusterRadius);
  for (unsigned int i = 0; i < _anisotropicClusters.size(); ++i) {
    dataCardinality->values()[i] = static_cast<double>(_anisotropicClusters[i].getNumberOfInputVertices());
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
  _anisotropicClusters.clear();
  // TODO: Don't reset this here
  _clusterRadius            = 0;
  this->_hasComputedMapping = false;
}

template <typename RADIAL_BASIS_FUNCTION_T>
std::string PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::getName() const
{
  return "partition-of-unity RBF";
}
} // namespace mapping
} // namespace precice
