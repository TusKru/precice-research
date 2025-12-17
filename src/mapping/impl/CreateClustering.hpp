#pragma once

#include <Eigen/Core>

#include <algorithm>

#include "math/math.hpp"
#include "mesh/Mesh.hpp"
#include "precice/impl/Types.hpp"
#include "query/Index.hpp"
// required for the squared distance computation
#include "mapping/RadialBasisFctSolver.hpp"

namespace precice::mapping::impl {

using Vertices = std::vector<mesh::Vertex>;

/**
 * This file contains helper functions for the clustering of a given mesh as required for the
 * partition of unity data mapping class.
 */

// Internal helper functions
namespace {

/**
 * @brief Transforms a multi-dimensional index into a unique index. The formula here corresponds to a
 * dimension-major scheme. The index is used to put all generated cluster centers into a unique and
 * and accessible place. The following functions @ref zCurveNeighborOffsets also allow to immediately
 * access (spatially) adjacent clusters.
 *
 * @param[in] ids The multi-dimensional index of the cluster center (x, y, z)
 * @param[in] nCells The maximum size of each index (x_max, y_max, z_max)
 * @return VertexID The serialized unique ID
 */
constexpr VertexID zCurve(std::array<unsigned int, 3> ids, std::array<unsigned int, 3> nCells)
{
  return (ids[0] * nCells[1] + ids[1]) * nCells[2] + ids[2];
}

// Overload to handle the offsets (via ints) properly
constexpr VertexID zCurve(std::array<int, 3> ids, std::array<unsigned int, 3> nCells)
{
  return (ids[0] * nCells[1] + ids[1]) * nCells[2] + ids[2];
}

/**
 * @brief Computes the index offsets of all spatially adjacent cluster centers
 *
 * @tparam dim spatial dimension of the multi-dimensional index
 * @param nCells the maximum size of each index (x_max, y_max, z_max)
 * @return The serialized unique neighbor ids
 */
template <int dim>
std::array<int, math::pow_int<dim>(3) - 1> zCurveNeighborOffsets(std::array<unsigned int, 3> nCells);

/// 2D implementation
template <>
std::array<int, 8> zCurveNeighborOffsets<2>(std::array<unsigned int, 3> nCells)
{
  // All  neighbor directions
  const std::array<std::array<int, 3>, 8> neighbors2DIndex = {{{-1, -1, 0}, {-1, 0, 0}, {-1, 1, 0}, {0, -1, 0}, {0, 1, 0}, {1, -1, 0}, {1, 0, 0}, {1, 1, 0}}};
  std::array<int, 8>                      neighbors2D{{}};
  // Uses the int overload of the zCurve
  // @todo: mark the function as constexpr, when moving to C++20 (transform is non-constexpr)
  std::transform(neighbors2DIndex.begin(), neighbors2DIndex.end(), neighbors2D.begin(), [&nCells](auto &v) { return zCurve(v, nCells); });

  return neighbors2D;
}

/// 3D implementation
template <>
std::array<int, 26> zCurveNeighborOffsets<3>(std::array<unsigned int, 3> nCells)
{
  // All  neighbor directions
  const std::array<std::array<int, 3>, 26> neighbors3DIndex = {{{-1, -1, -1}, {-1, -1, 0}, {-1, -1, 1}, {-1, 0, 0}, {-1, 0, 1}, {-1, 0, -1}, {-1, 1, 0}, {-1, 1, 1}, {-1, 1, -1}, {0, -1, -1}, {0, -1, 0}, {0, -1, 1}, {0, 0, 1}, {0, 0, -1}, {0, 1, 0}, {0, 1, 1}, {0, 1, -1}, {1, -1, -1}, {1, -1, 0}, {1, -1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 0, -1}, {1, 1, 0}, {1, 1, 1}, {1, 1, -1}}};
  std::array<int, 26>                      neighbors3D{{}};
  // Uses the int overload of the zCurve
  std::transform(neighbors3DIndex.begin(), neighbors3DIndex.end(), neighbors3D.begin(), [&nCells](auto &v) { return zCurve(v, nCells); });

  return neighbors3D;
}
/**
 * @brief This function computes all vertices (in \p vertices) within a sphere. The sphere is defined through a center vertex ( \p centerID ),
 * which needs to be part of the search space/vertex container itself and a \p radius . The function checks all vertices given in \p neighborOffsets.
 *
 * @tparam ArrayType static array of different sizes, depending on the spatial dimension
 *
 * @param[in] vertices The vertex container containing the center vertex ( \p centerID ) and the search space
 * @param[in] centerID ID of the center vertex defining the sphere
 * @param[in] radius radius of the sphere
 * @param[in] neighborOffsets Index offsets of adjacent vertices (depends only on the spatial dimension, see also \ref zCurveNeighborOffsets())
 *
 * @return std::vector<VertexID> a collection of vertexIDs (referring again to \p vertices ) of vertices within the sphere.
 */
template <typename ArrayType>
std::vector<VertexID> getNeighborsWithinSphere(const Vertices &vertices, VertexID centerID, double radius, const ArrayType neighborOffsets)
{
  // number of neighbors = (3^dim)-1
  static_assert((neighborOffsets.size() == 26) || (neighborOffsets.size() == 8));
  std::vector<VertexID> result;
  for (auto off : neighborOffsets) {
    // corner case where we have 'dead axis' in the bounding box
    if (off == 0) {
      continue;
    }
    // compute neighbor ID
    // @todo target type should be an unsigned int
    PRECICE_ASSERT(off != 0);
    // Assumption, vertex position in the container == vertex ID
    VertexID neighborID = centerID + off;
    // We might run out of the array bounds along the edges, so we only consider vertices inside
    if (neighborID >= 0 && neighborID < static_cast<int>(vertices.size())) {
      auto &neighborVertex = vertices[neighborID];
      auto &center         = vertices[centerID];
      if (!neighborVertex.isTagged() && (computeSquaredDifference(center.rawCoords(), neighborVertex.rawCoords()) < math::pow_int<2>(radius))) {
        result.emplace_back(neighborID);
      }
    }
  }
  return result;
}

/**
 * @brief Given vertices in the \p clusterCenters and a \p clusterRadius , this function
 * tags vertices in the \p clusterCenters, which don't contain any vertex of the \p mesh
 *
 * @param[in] clusterCenters vertices representing the cluster centers
 * @param[in] clusterRadius radius of the clusters
 * @param[in] mesh Mesh, in which we look for vertices
 */
void tagEmptyClusters(Vertices &clusterCenters, double clusterRadius, mesh::PtrMesh mesh)
{
  // Alternative implementation: mesh->index().getVerticesInsideBox() == 0
  std::for_each(clusterCenters.begin(), clusterCenters.end(), [&](auto &v) {
    if (!v.isTagged() && !mesh->index().isAnyVertexInsideBox(v, clusterRadius)) {
      v.tag();
    }
  });
}

/**
 * @brief Projects non-tagged cluster centers to the closest vertices of the given \p mesh
 *
 * @param[in] clusterCenters the collection of cluster centers we want to project
 * @param[in] mesh the mesh we search for the closest vertex in order to move the center
 *
 * @note This function preserves the size of the \p clusterCenters , as tagged vertices remain unchanged.
 */
void projectClusterCentersToinputMesh(Vertices &clusterCenters, mesh::PtrMesh mesh)
{
  std::transform(clusterCenters.begin(), clusterCenters.end(), clusterCenters.begin(), [&](auto &v) {
    if (!v.isTagged()) {
      auto closestCenter = mesh->index().getClosestVertex(v.getCoords()).index;
      return mesh::Vertex{mesh->vertex(closestCenter).getCoords(), v.getID()};
    } else {
      return v;
    }
  });
}

/**
 * @brief Tag (non-tagged) duplicated vertices in the \p centers , where duplicate is measured as a distance
 * between vertices smaller than a \p threshold . Here, we use the static neighbors from our index curve
 * (see also \ref zCurve and \ref zCurveNeighborOffsets ).
 *
 * @tparam dim spatial dimension of the clustering
 *
 * @param[in] centers vertex container, where we look for duplicate vertices
 * @param[in] nCells the maximum size of each index (x_max, y_max, z_max)
 * @param[in] threshold threshold value, which compares against the distance between centers
 */
template <int dim>
void tagDuplicateCenters(Vertices &centers, std::array<unsigned int, 3> nCells, double threshold)
{
  PRECICE_ASSERT(threshold >= 0);
  if (centers.empty())
    return;

  // Get the index offsets for the z curve
  auto neighborOffsets = zCurveNeighborOffsets<dim>(nCells);
  static_assert((neighborOffsets.size() == 8 && dim == 2) || (neighborOffsets.size() == 26 && dim == 3));

  // For the following to work, the vertexID has to correspond to the position in the centers
  PRECICE_ASSERT(std::all_of(centers.begin(), centers.end(), [idx = 0](auto &v) mutable { return v.getID() == idx++; }));
  // we check all neighbors
  for (auto &v : centers) {
    if (!v.isTagged()) {
      auto ids = getNeighborsWithinSphere(centers, v.getID(), threshold, neighborOffsets);
      // @todo check more selective which one to remove
      if (ids.size() > 0)
        v.tag();
    }
  }
}

/**
 * @brief Removes and erases tagged vertices from the input \p container.
 *
 * @param[in] container vertex container
 *
 * @note The neighbor search as carried out above (see \ref tagDuplicateCenters) won't work after calling this function,
 * as the position of vertices in the container changes.
 */
void removeTaggedVertices(Vertices &container)
{
  container.erase(std::remove_if(container.begin(), container.end(), [](auto &v) { return v.isTagged(); }), container.end());
}

/**
 * @brief Struct holding the radius, densities and sample points required for the clustering.
 * 
 * @note This struct is initialized in \ref estimateDensityField and used in \ref createClustering.
 */
struct DensityField {
  double                medianRadius;
  std::vector<VertexID> sampleIDs;
  std::vector<double>   sampleRadii;

  DensityField(double medRad,
               std::vector<VertexID> samples,
               std::vector<double>   radii)
      : medianRadius(medRad),
        sampleIDs(std::move(samples)),
        sampleRadii(std::move(radii))
  {
    PRECICE_ASSERT(sampleIDs.size() == sampleRadii.size(), "Inconsistent sizes in DensityField struct initialization.");
  }
};

// 子区域数据结构：存储子区域边界、对应取样半径、局部簇间距
struct SubRegion {
  precice::mesh::BoundingBox subBB;       // 子区域边界框
  double sampleRadius{0.0};      // 子区域对应的密度取样半径（反映局部网格密度）
  std::vector<double> distance;  // 子区域内的簇中心间距（各维度）

  SubRegion(const precice::mesh::BoundingBox &bb,
            double                           radius,
            const std::vector<double>       &dist)
      : subBB(bb),
        sampleRadius(radius),
        distance(dist)
  {}
};
} // namespace

/**
 * @brief Computes an estimate for the cluster radius, which results in approximately \p verticesPerCluster vertices inside
 * of each cluster. The algorithm generates random samples in the domain and queries the \p verticesPerCluster nearest-neighbors
 * from the mesh index tree. The cluster radius is then estimated through the distance between the center vertex (random sample)
 * and the vertex the furthest away from the center (being on the edge of the cluster).
 *
 * @param[in] verticesPerCluster target number of vertices in each cluster.
 * @param[in] inMesh mesh we want to create the clustering on
 * @param[in] bb bounding box of the domain. Used to place the random samples in the domain.
 *
 * @return The estimate for the cluster radius.
 */
inline double estimateClusterRadius(unsigned int verticesPerCluster, mesh::PtrMesh inMesh, const precice::mesh::BoundingBox &bb)
{
/*
1. **生成随机样本**
   - 先将边界框（`bb`）中心最近的网格顶点作为首个样本；
   - 针对网格的每个空间维度，在边界框中心上下各0.25倍边长位置生成样本，再将所有样本“移动”到输入网格（`inMesh`）中最近的顶点，避免壳状网格出现空簇。
2. **计算样本聚类半径**
   - 对每个随机样本（作为聚类中心），从网格索引树查询其`verticesPerCluster`个最近邻顶点；
   - 计算中心到各最近邻顶点的平方距离，取最大值开方，作为该样本的聚类半径。
3. **确定最终聚类半径**
   - 对所有样本的聚类半径排序，取中位数作为最终的聚类半径估算值（前提是样本数为奇数，确保中位数有效）。
*/

  // Step 1: Generate random samples from the input mesh
  // 步骤1：从输入网格生成随机样本
  std::vector<VertexID> randomSamples;

  PRECICE_ASSERT(!bb.isDefault(), "Invalid bounding box.");
  // All samples are 'moved' to the closest vertex from the input mesh in order
  // to avoid empty clusters when we have shell-like meshes (as in surface coupling)
  // The first sample: the vertex closest to the bounding box center
  // 1. 所有样本均会“移动”至输入网格中距离自身最近的顶点，目的是避免在处理壳状网格（如表面耦合场景）时出现空簇。
  // 2. 第一个样本的定位规则：选择距离边界框中心最近的顶点作为其移动目标。
  const auto bbCenter = bb.center();
  randomSamples.emplace_back(inMesh->index().getClosestVertex(bbCenter).index);

  // Now we place samples for each space dimension above and below the bounding box center
  // and look for the closest vertex in the input mesh
  // In 2D the sampling would like this
  // 1. 核心操作：针对每个空间维度，在边界框中心的上下（或前后/左右，依维度而定）位置放置样本。
  // 2. 目的：在输入的网格模型中查找与这些样本距离最近的顶点。
  // 3. 补充说明：该采样方式在二维（2D）场景下的具体表现可参考后续（未展示的）相关说明或示例。
  // +---------------------+
  // |          x          |
  // |                     |
  // |    x     c      x   |
  // |                     |
  // |          x          |
  // +---------------------+
  for (int d = 0; d < inMesh->getDimensions(); ++d) {
    auto sample = bbCenter;
    // Lower as the center
    sample[d] -= bb.getEdgeLength(d) * 0.25;
    randomSamples.emplace_back(inMesh->index().getClosestVertex(sample).index);
    // Higher than the center
    sample[d] += bb.getEdgeLength(d) * 0.5;
    randomSamples.emplace_back(inMesh->index().getClosestVertex(sample).index);
  }

  // Step 2: Compute the radius of the randomSamples ('centers'), which would have verticesPerCluster vertices
  // 步骤2：计算随机样本（“中心”）的半径，每个样本将有verticesPerCluster个顶点
  std::vector<double> sampledClusterRadii;
  for (auto s : randomSamples) {
    // ask the index tree for the k-nearest neighbors in order to estimate the point density
    auto kNearestVertexIDs = inMesh->index().getClosestVertices(inMesh->vertex(s).getCoords(), verticesPerCluster);
    // compute the distance of each point to the center
    std::vector<double> squaredRadius(kNearestVertexIDs.size());
    std::transform(kNearestVertexIDs.begin(), kNearestVertexIDs.end(), squaredRadius.begin(), [&inMesh, s](auto i) {
      return computeSquaredDifference(inMesh->vertex(i).rawCoords(), inMesh->vertex(s).rawCoords());
    });
    // Store the maximum distance
    auto maxRadius = std::max_element(squaredRadius.begin(), squaredRadius.end());
    sampledClusterRadii.emplace_back(std::sqrt(*maxRadius));
  }

  // Step 3: Sort (using nth_element) the sampled radii and select the median as cluster radius
  PRECICE_ASSERT(sampledClusterRadii.size() % 2 != 0, "Median calculation is only valid for odd number of elements.");
  PRECICE_ASSERT(sampledClusterRadii.size() > 0);
  unsigned int middle = sampledClusterRadii.size() / 2;
  std::nth_element(sampledClusterRadii.begin(), sampledClusterRadii.begin() + middle, sampledClusterRadii.end());
  double clusterRadius = sampledClusterRadii[middle];

  return clusterRadius;
}

/**
 * @brief Computes an estimate for the cluster density, which results in approximately \p verticesPerCluster vertices inside
 * of each cluster. The algorithm generates random samples in the domain and queries the \p verticesPerCluster nearest-neighbors
 * from the mesh index tree. The cluster density is then estimated through the distance between the center vertex (random sample)
 * and the vertex the furthest away from the center (being on the edge of the cluster).
 *
 * @param[in] verticesPerCluster target number of vertices in each cluster.
 * @param[in] inMesh mesh we want to create the clustering on
 * @param[in] bb bounding box of the domain. Used to place the random samples in the domain.
 *
 * @return The density in vector for all the samples.
 */
// inline struct DensityField estimateDensityField(unsigned int verticesPerCluster, mesh::PtrMesh inMesh, const precice::mesh::BoundingBox &bb)
// {
//   // Step 1: Generate random samples from the input mesh
//   // 步骤1：从输入网格生成随机样本
//   std::vector<VertexID> randomSamples;

//   PRECICE_ASSERT(!bb.isDefault(), "Invalid bounding box.");
//   // All samples are 'moved' to the closest vertex from the input mesh in order
//   // to avoid empty clusters when we have shell-like meshes (as in surface coupling)
//   // The first sample: the vertex closest to the bounding box center
//   // 1. 所有样本均会“移动”至输入网格中距离自身最近的顶点，目的是避免在处理壳状网格（如表面耦合场景）时出现空簇。
//   // 2. 第一个样本的定位规则：选择距离边界框中心最近的顶点作为其移动目标。
//   const auto bbCenter = bb.center();
//   randomSamples.emplace_back(inMesh->index().getClosestVertex(bbCenter).index);

//   // Now we place samples for each space dimension above and below the bounding box center
//   // and look for the closest vertex in the input mesh
//   // In 2D the sampling would like this
//   // 1. 核心操作：针对每个空间维度，在边界框中心的上下（或前后/左右，依维度而定）位置放置样本。
//   // 2. 目的：在输入的网格模型中查找与这些样本距离最近的顶点。
//   // 3. 补充说明：该采样方式在二维（2D）场景下的具体表现可参考后续（未展示的）相关说明或示例。
//   // 
//   // ^
//   // |---------------------+
//   // |    x    ->     x    |
//   // |                     |
//   // |    ^     c     v    |
//   // |                     |
//   // |    x           x    |
//   // +------------------------>
//   if (inMesh->getDimensions() == 2){
//     auto sample = bbCenter;
//     sample[0] -= bb.getEdgeLength(0) * 0.25;
//     sample[1] -= bb.getEdgeLength(1) * 0.25;
//     randomSamples.emplace_back(inMesh->index().getClosestVertex(sample).index);
//     sample[1] += bb.getEdgeLength(1) * 0.5;
//     randomSamples.emplace_back(inMesh->index().getClosestVertex(sample).index);
//     sample[0] += bb.getEdgeLength(0) * 0.5;
//     randomSamples.emplace_back(inMesh->index().getClosestVertex(sample).index);
//     sample[1] -= bb.getEdgeLength(1) * 0.5;
//     randomSamples.emplace_back(inMesh->index().getClosestVertex(sample).index);
//   } else if (inMesh->getDimensions() == 3){
//     auto sample = bbCenter;
//     sample[0] -= bb.getEdgeLength(0) * 0.25;
//     sample[1] -= bb.getEdgeLength(1) * 0.25;
//     sample[2] -= bb.getEdgeLength(2) * 0.25;
//     randomSamples.emplace_back(inMesh->index().getClosestVertex(sample).index);
//     sample[1] += bb.getEdgeLength(1) * 0.5;
//     randomSamples.emplace_back(inMesh->index().getClosestVertex(sample).index);
//     sample[0] += bb.getEdgeLength(0) * 0.5;
//     randomSamples.emplace_back(inMesh->index().getClosestVertex(sample).index);
//     sample[1] -= bb.getEdgeLength(1) * 0.5;
//     randomSamples.emplace_back(inMesh->index().getClosestVertex(sample).index);
//     sample[2] += bb.getEdgeLength(2) * 0.5;
//     randomSamples.emplace_back(inMesh->index().getClosestVertex(sample).index);
//     sample[0] -= bb.getEdgeLength(0) * 0.5;
//     randomSamples.emplace_back(inMesh->index().getClosestVertex(sample).index);
//     sample[1] += bb.getEdgeLength(1) * 0.5;
//     randomSamples.emplace_back(inMesh->index().getClosestVertex(sample).index);
//     sample[0] += bb.getEdgeLength(0) * 0.5;
//     randomSamples.emplace_back(inMesh->index().getClosestVertex(sample).index);
//   }
  
//   // Step 2: Compute the radius of the randomSamples ('centers'), which would have verticesPerCluster vertices
//   // 步骤2：计算随机样本（“中心”）的半径，每个样本将有verticesPerCluster个顶点，再据此计算密度
//   // @todo: maybe sort the densities or select the median/max/min as cluster density
//   std::vector<double> sampleRadii;
//   for (auto s : randomSamples) {
//     // ask the index tree for the k-nearest neighbors in order to estimate the point density
//     auto kNearestVertexIDs = inMesh->index().getClosestVertices(inMesh->vertex(s).getCoords(), verticesPerCluster);
//     // compute the distance of each point to the center
//     std::vector<double> squaredRadius(kNearestVertexIDs.size());
//     std::transform(kNearestVertexIDs.begin(), kNearestVertexIDs.end(), squaredRadius.begin(), [&inMesh, s](auto i) {
//       return computeSquaredDifference(inMesh->vertex(i).rawCoords(), inMesh->vertex(s).rawCoords());
//     });
//     // Store the maximum distance
//     auto maxRadius = std::max_element(squaredRadius.begin(), squaredRadius.end());
//     double radius = std::sqrt(*maxRadius);
//     sampleRadii.emplace_back(radius);
//   }

//   // Step 3: Sort (using nth_element) the sampled radii and select the median as cluster radius
//   PRECICE_ASSERT(sampleRadii.size() % 2 != 0, "Median calculation is only valid for odd number of elements.");
//   PRECICE_ASSERT(sampleRadii.size() > 0);
//   unsigned int middle = sampleRadii.size() / 2;
//   std::nth_element(sampleRadii.begin(), sampleRadii.begin() + middle, sampleRadii.end());
//   double medianRadius = sampleRadii[middle];

//   // Step 4: Return the struct holding all density field information
//   return DensityField(medianRadius, randomSamples, sampleRadii);
// }

/**
 * 使用最远点采样(FPS)算法自适应选择簇中心点
 * @param inMesh 输入网格
 * @param outMesh 输出网格
 * @param relativeOverlap 相邻簇的相对重叠度
 * @param verticesPerCluster 每个簇的目标顶点数
 * @param projectClustersToInput 是否将簇中心投影到输入网格
 * @return 包含簇中心集合与每个簇对应半径的元组
 */
inline std::tuple<Vertices, std::vector<double>> selectAdaptiveClusterCenters(
    mesh::PtrMesh inMesh,
    mesh::PtrMesh outMesh,
    double        relativeOverlap,
    unsigned int  verticesPerCluster,
    bool          projectClustersToInput)
{
  // 基本断言：维度一致、参数合法
  PRECICE_ASSERT(inMesh->getDimensions() == outMesh->getDimensions());
  PRECICE_ASSERT(verticesPerCluster > 0);
  PRECICE_ASSERT(relativeOverlap < 1.0);
  const int inDim = inMesh->getDimensions();

  // 边界情况：输入网格为空或输出网格为空且非JIT，直接返回空结果
  if (inMesh->empty() || (outMesh->empty() && !outMesh->isJustInTime())) {
    return {Vertices{}, std::vector<double>{}};
  }

  // 初始中心：选择包围盒中心最近的顶点作为第一个中心
  const auto bb = inMesh->index().getRtreeBounds();
  const auto initialID = inMesh->index().getClosestVertex(bb.center()).index;

  Vertices centers;
  std::vector<double> radii;
  std::vector<double> centerCoords(inDim);

  // 将初始中心写入集合，半径后续估计
  auto rawCoords = inMesh->vertex(initialID).rawCoords();
  for (int d = 0; d < inDim; ++d) {
    centerCoords[d] = rawCoords[d];
  }
  centers.emplace_back(mesh::Vertex({centerCoords, static_cast<int>(centers.size())}));
  radii.emplace_back(0.0);

  // 维护每个顶点到最近已选中心的最小距离（FPS的核心数据）
  const std::size_t nV = inMesh->nVertices();
  std::vector<double> minDistSq(nV, std::numeric_limits<double>::infinity());
  // std::vector<int>    nearestCenter(nV, -1);
  std::vector<bool> isCovered(nV, false);

  // 局部半径估计：查询k近邻，以第k近邻的距离作为簇半径
  auto estimateRadiusAndCover = [&](const mesh::Vertex &v) {
    auto ids = inMesh->index().getClosestVertices(v.getCoords(), static_cast<int>(verticesPerCluster));
    for (auto id : ids) {
      isCovered[static_cast<size_t>(id)] = true;
    }
    std::vector<double> sq(ids.size());
    std::transform(ids.begin(), ids.end(), sq.begin(), [&](auto i) {
      return computeSquaredDifference(inMesh->vertex(i).rawCoords(), v.rawCoords());
    });
    auto it = std::max_element(sq.begin(), sq.end());
    return it != sq.end() ? std::sqrt(*it) : 0.0;
  };

  // 计算第一个中心的半径
  radii[0] = estimateRadiusAndCover(centers[0]);

  for (std::size_t vid = 0; vid < nV; ++vid) {
    const auto &v = inMesh->vertex(static_cast<int>(vid));
    double d2     = computeSquaredDifference(v.rawCoords(), centers[0].rawCoords());
    minDistSq[vid] = d2;
    // nearestCenter[vid] = 0;
  }

  // 顶点覆盖判定：若到最近中心的距离小于阈值则视为已覆盖
  // auto isCovered = [&](std::size_t vid) {
  //   int cidx = nearestCenter[vid];
  //   // 基于半径与相对重叠度计算每个中心的覆盖阈值（平方）
  //   return cidx >= 0 && minDistSq[vid] <= math::pow_int<2>(radii[cidx] * (1.0 - relativeOverlap));
  // };

  // 目标簇数量：按总顶点数与每簇目标顶点数估算
  // const std::size_t targetClusters = static_cast<unsigned int>(std::ceil(static_cast<double>(nV) / static_cast<double>(verticesPerCluster)));

  // FPS迭代：每次选择未覆盖集合中距离当前中心集合最远的顶点作为新中心
  bool continueIteration = true;
  while (continueIteration) {
    std::size_t farthestID = nV;
    double      farthestVal = -1.0;
    for (std::size_t vid = 0; vid < nV; ++vid) {
      if (!isCovered[vid]) {
        if (minDistSq[vid] > farthestVal) {
          farthestVal = minDistSq[vid];
          farthestID  = vid;
        }
      }
    }

    // 若没有未覆盖顶点，提前终止
    if (farthestID == nV) {
      continueIteration = false;
      break;
    }

    // 新中心加入并估计其局部半径
    const auto &vf = inMesh->vertex(static_cast<int>(farthestID));
    for (int d = 0; d < inDim; ++d) {
      centerCoords[d] = vf.rawCoords()[d];
    }
    centers.emplace_back(mesh::Vertex({centerCoords, static_cast<int>(centers.size())}));
    radii.emplace_back(estimateRadiusAndCover(centers.back()));

    // 更新所有顶点到最近中心的最小距离与对应中心索引
    // @todo 若最小距离向量和最近中心索引向量无其他用途，此处可不更新已经被覆盖的顶点
    for (std::size_t vid = 0; vid < nV; ++vid) {
      const auto &v = inMesh->vertex(static_cast<int>(vid));
      double d2     = computeSquaredDifference(v.rawCoords(), centers.back().rawCoords());
      if (d2 < minDistSq[vid]) {
        minDistSq[vid]    = d2;
        // nearestCenter[vid] = static_cast<int>(centers.size() - 1);
      }
    }

    continueIteration = false;
    for (std::size_t vid = 0; vid < nV; ++vid) {
      if (!isCovered[vid]) {
        continueIteration = true;
        break;
      }
    }
    // 半径重叠自适应调整：确保邻近簇满足重叠要求
    // adjustRadiiForOverlap(centers, radii, inMesh, relativeOverlap);
  }

  // 可选：将中心投影到输入网格最近顶点，增强不规则/壳状网格的稳健性
  // if (projectClustersToInput) {
  //   projectClusterCentersToinputMesh(centers, inMesh);
  // }

  return {centers, radii};
}

/**
 * 调整簇半径以确保相邻簇有足够重叠
 * @param centers 簇中心顶点ID向量
 * @param radii 簇半径向量
 * @param mesh 输入网格
 * @param relativeOverlap 要求的相对重叠度
 */
inline void adjustRadiiForOverlap(
    std::vector<mesh::Vertex> &centers,
    std::vector<double>       &radii,
    mesh::PtrMesh              mesh,
    double                     relativeOverlap)
{
  PRECICE_ASSERT(centers.size() == radii.size());
  PRECICE_ASSERT(relativeOverlap < 1.0);
  if (centers.empty()) return;

  // 将中心点构造成临时网格，建立R树索引以支持邻近查询
  mesh::Mesh tempMesh{"cluster-centers", mesh->getDimensions(), mesh::Mesh::MESH_ID_UNDEFINED};
  for (const auto &c : centers) {
    tempMesh.createVertex(c.getCoords());
  }
  query::Index centerIndex(tempMesh);

  // 迭代调整：逐对检查相邻中心是否满足重叠约束，不满足则按比例增大半径
  const double eps = 1e-12;
  int          maxIter = 50;
  while (maxIter-- > 0) {
    bool changed = false;
    // 计算当前最大半径，用于构造邻近查询的盒子大小
    double maxR = 0.0;
    for (double r : radii) maxR = std::max(maxR, r);
    for (std::size_t i = 0; i < centers.size(); ++i) {
      // 使用R树在以(i)为中心、半径(r_i + maxR)的盒子内查找可能的相邻中心
      auto neighbors = centerIndex.getVerticesInsideBox(tempMesh.vertex(static_cast<int>(i)), radii[i] + maxR);
      for (auto jID : neighbors) {
        std::size_t j = static_cast<std::size_t>(jID);
        if (j == i) continue;
        // 计算两中心距离与重叠阈值
        double d2 = computeSquaredDifference(centers[i].rawCoords(), centers[j].rawCoords());
        double d  = std::sqrt(d2);
        double threshold = (radii[i] + radii[j]) * (1.0 - relativeOverlap);
        // 若不满足 d < threshold，则需要增大两半径至满足重叠
        if (!(d < threshold - eps)) {
          double requiredSum = d / (1.0 - relativeOverlap);
          double deficit     = requiredSum - (radii[i] + radii[j]);
          if (deficit > 0) {
            // 按比例同时放大两个簇的半径，缩放因子alpha与缺口占当前半径和之比成正比
            double alpha = deficit / std::max(eps, (radii[i] + radii[j]));
            radii[i] *= (1.0 + alpha);
            radii[j] *= (1.0 + alpha);
            changed = true;
          }
        }
      }
    }
    // 若本轮没有变化，说明已满足重叠约束，终止迭代
    if (!changed) break;
  }
}

/**
 * @brief Creates a clustering as a collection of Vertices (representing the cluster centers) and a cluster radius,
 * as required for the partition of unity mapping. The algorithm estimates a cluster radius based on the input parameter
 * \p verticesPerCluster (see also \ref estimateClusterRadius above, which is directly used by the function). Afterwards,
 * the algorithm creates a cartesian-like grid of center vertices, where the distance of the centers is defined through
 * the \p relativeOverlap and the cluster radius. The parameter \p projectClustersToInput moves the cartesian center
 * vertices to the closest vertex from the input mesh, which is useful in case of very irregular meshes or shell-shaped
 * meshes.
 * The algorithm also removes potentially empty cluster, i.e., clusters which would have either no vertex from the
 * \p inMesh or from the \p outMesh . See also \ref tagEmptyClusters.
 *
 * @param[in] inMesh The input mesh (input mesh for consistent, output mesh for conservative mappings), on which the
 *            clustering is computed. The input parameters \p verticesPerCluster and \p projectClustersToInput refer
 *            to the \p inMesh
 * @param[in] outMesh The output mesh (output mesh for consistent, input mesh for conservative mappings),
 * @param[in] relativeOverlap Value between zero and one, which steers the relative distance between cluster centers.
 *            A value of zero leads to no overlap, a value of one would lead to a complete overlap between clusters.
 * @param[in] verticesPerCluster Target number of vertices per partition.
 * @param[in] projectClustersToInput if enabled, moves the cluster centers to the closest vertex of the \p inMesh
 *
 * @return a tuple for the cluster radius and a vector of vertices marking the cluster centers
 */
inline std::tuple<double, Vertices> createClustering(mesh::PtrMesh inMesh, mesh::PtrMesh outMesh,
                                                     double relativeOverlap, unsigned int verticesPerCluster,
                                                     bool projectClustersToInput)
{
  precice::logging::Logger _log{"impl::createClustering"};
  PRECICE_TRACE();
  PRECICE_ASSERT(relativeOverlap < 1);
  PRECICE_ASSERT(verticesPerCluster > 0);
  PRECICE_ASSERT(inMesh->getDimensions() == outMesh->getDimensions());

  // If we have either no input or no output vertices, we return immediately
  if (inMesh->empty() || (outMesh->empty() && !outMesh->isJustInTime())) {
    return {double{}, Vertices{}};
  }

  PRECICE_ASSERT(!inMesh->empty());
  PRECICE_ASSERT(!outMesh->empty() || outMesh->isJustInTime());
  // startGridAtEdge boolean switch in order to decide either to start the clustering at the edge of the bounding box in each direction
  // (true) or start the clustering inside the bounding box (edge + 0.5 radius). The latter approach leads to fewer clusters,
  // but might result in a worse clustering
  const bool startGridAtEdge = projectClustersToInput;

  PRECICE_DEBUG("Relative overlap: {}", relativeOverlap);
  PRECICE_DEBUG("Vertices per cluster: {}", verticesPerCluster);

  // Step 1: Compute the local bounding box of the input mesh manually
  // Note that we don't use the corresponding bounding box functions from
  // precice::mesh (e.g. ::getBoundingBox), as the stored bounding box might
  // have the wrong size (e.g. direct access)
  // @todo: Which mesh should be used in order to determine the cluster centers:
  // pro outMesh: we want perfectly fitting clusters around our output vertices
  // however, this makes the cluster distribution/mapping dependent on the output
  precice::mesh::BoundingBox localBB = inMesh->index().getRtreeBounds();

#ifndef NDEBUG
  // Safety check
  precice::mesh::BoundingBox bb_check(inMesh->getDimensions());
  for (const mesh::Vertex &vertex : inMesh->vertices()) {
    bb_check.expandBy(vertex);
  }
  PRECICE_ASSERT(bb_check == localBB);
#endif

  // If we have very few vertices in the domain, (in this case twice our cluster
  // size as we decompose most probably at least in 4 clusters) we just use a
  // single cluster. The clustering result of the algorithm further down is in
  // this case not optimal and might lead to too many clusters.
  // The single cluster has in principle a radius of inf. We use here twice the
  // length of the longest bounding box edge length and the center of the bounding
  // box for the center point.
  if (inMesh->nVertices() < verticesPerCluster * 2) {
    double cRadius = localBB.longestEdgeLength();
    if (cRadius == 0) {
      PRECICE_WARN("Determining a cluster radius failed. This is most likely a result of a coupling mesh consisting of just a single vertex. Setting the cluster radius to 1.");
      cRadius = 1;
    }
    return {cRadius * 2, Vertices{mesh::Vertex({localBB.center(), 0})}};
  }
  // We define a convenience alias for the localBB. In case we need to synchronize the clustering across ranks later on, we need
  // to work with the global bounding box of the whole domain.
  auto globalBB = localBB;

  // Step 2: Now we pick random samples from the input mesh and ask the index tree for the k-nearest neighbors
  // in order to estimate the point density and determine a proper cluster radius (see also the function documentation)
  double clusterRadius = estimateClusterRadius(verticesPerCluster, inMesh, localBB);
  PRECICE_DEBUG("Vertex cluster radius: {}", clusterRadius);

  // maximum distance between cluster centers lying diagonal to each other. The maximum distance takes the overlap condition into
  // account: example for 2D: if the distance between the centers is sqrt( 4 / 2 ) * radius, we violate the overlap condition between
  // diagonal clusters
  const int    inDim                 = inMesh->getDimensions();
  const double maximumCenterDistance = std::sqrt(4. / inDim) * clusterRadius * (1 - relativeOverlap);

  // Step 3: using the maximum distance and the bounding box, compute the number of clusters in each direction
  // we ceil the number of clusters in order to guarantee the desired overlap
  std::array<unsigned int, 3> nClustersGlobal{1, 1, 1};
  for (int d = 0; d < globalBB.getDimension(); ++d)
    nClustersGlobal[d] = std::ceil(std::max(1., globalBB.getEdgeLength(d) / maximumCenterDistance));

  // Step 4: Determine the centers of the clusters
  Vertices centers;
  // Vector used to temporarily store each center coordinates
  std::vector<double> centerCoords(inDim);
  // Vector storing the distances between the cluster in each direction
  std::vector<double> distances(inDim);
  // Vector storing the starting coordinates in each direction
  std::vector<double> start(inDim);
  // Fill the constant vectos, i.e., the distances and the start
  for (unsigned int d = 0; d < distances.size(); ++d) {
    // this distance calculation guarantees that we have at least the minimal specified overlap, as we ceil the division for the number of clusters
    // as an alternative, once could use distance = const = maximumCenterDistance, which might lead to better results when projecting and removing duplicates (?)
    distances[d] = globalBB.getEdgeLength(d) / (nClustersGlobal[d]);
    start[d]     = globalBB.minCorner()(d);
  }
  PRECICE_DEBUG("Distances: {}", distances);

  // Step 5: Take care of the starting layer: if we start the grid at the globalBB edge we have an additional layer in each direction
  if (startGridAtEdge) {
    for (int d = 0; d < inDim; ++d) {
      if (globalBB.getEdgeLength(d) > math::NUMERICAL_ZERO_DIFFERENCE) {
        nClustersGlobal[d] += 1;
      }
    }
  } // If we don't start at the edge, we shift the starting layer into the middle by 0.5 * distance
  else {
    std::transform(start.begin(), start.end(), distances.begin(), start.begin(), [](auto s, auto d) { return s + 0.5 * d; });
  }

  // Print some information
  PRECICE_DEBUG("Global cluster distribution: {}", nClustersGlobal);
  unsigned int nTotalClustersGlobal = std::accumulate(nClustersGlobal.begin(), nClustersGlobal.end(), 1U, std::multiplies<unsigned int>());
  PRECICE_DEBUG("Global number of total clusters (tentative): {}", nTotalClustersGlobal);

  // Step 6: transform the global metrics (number of clusters and starting layer) into local ones
  // Since the local and global bounding boxes are the same in the current implementation, the
  // global metrics and the local metrics will be the same.
  // First, we need to update the starting points of our clustering scheme
  std::array<unsigned int, 3> nClustersLocal{1, 1, 1};
  for (unsigned int d = 0; d < start.size(); ++d) {
    // Exclude the case where we have no distances (nClustersGlobal[d] == 1 )
    if (distances[d] > 0) {
      start[d] += std::ceil(((localBB.minCorner()(d) - start[d]) / distances[d]) - math::NUMERICAL_ZERO_DIFFERENCE) * distances[d];
      // One cluster for the starting point and a further one for each distance we can fit into the BB
      nClustersLocal[d] = 1 + std::floor(((localBB.maxCorner()(d) - start[d]) / distances[d]) + math::NUMERICAL_ZERO_DIFFERENCE);
    }
  }

  unsigned int nTotalClustersLocal = std::accumulate(nClustersLocal.begin(), nClustersLocal.end(), 1U, std::multiplies<unsigned int>());
  centers.resize(nTotalClustersLocal, mesh::Vertex({centerCoords, -1}));

  // Print some information
  PRECICE_DEBUG("Local cluster distribution: {}", nClustersLocal);
  PRECICE_DEBUG("Local number of total clusters (tentative): {}", nTotalClustersLocal);

  // Step 7: fill the center container using the zCurve for indexing. For the further processing, it is also important to ensure
  // that the ID within the center Vertex class coincides with the position of the center vertex in the container.
  // We start with the (bottom left) corner and iterate over all dimension
  PRECICE_ASSERT(inDim >= 2);
  if (inDim == 2) {
    centerCoords[0] = start[0];
    for (unsigned int x = 0; x < nClustersLocal[0]; ++x, centerCoords[0] += distances[0]) {
      centerCoords[1] = start[1];
      for (unsigned int y = 0; y < nClustersLocal[1]; ++y, centerCoords[1] += distances[1]) {
        auto id = zCurve(std::array<unsigned int, 3>{{x, y, 0}}, nClustersLocal);
        PRECICE_ASSERT(id < static_cast<int>(centers.size()) && id >= 0);
        centers[id] = mesh::Vertex({centerCoords, id});
      }
    }
  } else {
    centerCoords[0] = start[0];
    for (unsigned int x = 0; x < nClustersLocal[0]; ++x, centerCoords[0] += distances[0]) {
      centerCoords[1] = start[1];
      for (unsigned int y = 0; y < nClustersLocal[1]; ++y, centerCoords[1] += distances[1]) {
        centerCoords[2] = start[2];
        for (unsigned int z = 0; z < nClustersLocal[2]; ++z, centerCoords[2] += distances[2]) {
          auto id = zCurve(std::array<unsigned int, 3>{{x, y, z}}, nClustersLocal);
          PRECICE_ASSERT(id < static_cast<int>(centers.size()) && id >= 0);
          centers[id] = mesh::Vertex({centerCoords, id});
        }
      }
    }
  }

  // Step 8: tag vertices, which should be removed later on

  // Step 8a: tag empty clusters
  tagEmptyClusters(centers, clusterRadius, inMesh);
  if (!outMesh->isJustInTime()) {
    tagEmptyClusters(centers, clusterRadius, outMesh);
  }
  PRECICE_DEBUG("Number of non-tagged centers after empty clusters were filtered: {}", std::count_if(centers.begin(), centers.end(), [](auto &v) { return !v.isTagged(); }));

  // Step 9: Move the cluster centers if desired
  if (projectClustersToInput) {
    projectClusterCentersToinputMesh(centers, inMesh);
    // @todo: The duplication should probably be defined in terms of the target distance, not the actual distance. Find good default values
    const auto duplicateThreshold = 0.4 * (*std::min_element(distances.begin(), distances.end()));
    PRECICE_DEBUG("Tagging duplicates using the threshold value {} for the regular distances {}", duplicateThreshold, distances);
    // usually inMesh->getDimensions() == 2
    // to also cover the case where we have only a single layer in a 3D computation we use the number of clusters in z direction
    // Step 8b or 9a: Moving cluster centers might lead to duplicate centers or centers being too close to each other. Therefore,
    // we tag duplicate centers (see also the documentation of \ref tagDuplicateCenters ).
    if (nClustersLocal[2] == 1) {
      tagDuplicateCenters<2>(centers, nClustersLocal, duplicateThreshold);
    } else {
      tagDuplicateCenters<3>(centers, nClustersLocal, duplicateThreshold);
    }
    // the tagging here won't filter out a lot of clusters, but finding them anyway allows us to allocate the right amount of
    // memory for the cluster vector in the mapping, which would otherwise be expensive
    // we cannot hit any empty vertices in the inputmesh
    if (!outMesh->isJustInTime()) {
      tagEmptyClusters(centers, clusterRadius, outMesh);
    }
    PRECICE_DEBUG("Number of non-tagged centers after duplicate tagging : {}", std::count_if(centers.begin(), centers.end(), [](auto &v) { return !v.isTagged(); }));
  }
  PRECICE_ASSERT(std::all_of(centers.begin(), centers.end(), [idx = 0](auto &v) mutable { return v.getID() == idx++; }));

  // Step 10: finally, remove the vertices from the center container
  removeTaggedVertices(centers);
  PRECICE_ASSERT(std::none_of(centers.begin(), centers.end(), [](auto &v) { return v.isTagged(); }));
  PRECICE_CHECK(centers.size() > 0, "Too many vertices have been filtered out.");

  return {clusterRadius, centers};
}

/**
 * @brief 不均匀簇生成函数：按子区域密度调整簇中心间距，返回簇中心+对应半径
 * @param[in] inMesh 基准网格（用于密度计算和可选投影）
 * @param[in] outMesh 关联网格（用于空簇判断）
 * @param[in] relativeOverlap 簇重叠系数（0-1，控制簇间距基准）
 * @param[in] verticesPerCluster 目标簇内顶点数（用于密度估算）
 * @param[in] projectClustersToInput 是否将簇中心投影到inMesh最近顶点
 * @return 子域簇半径向量（数量与子域数量相等） + 簇中心向量
 */
// inline std::tuple<std::vector<unsigned int>, std::vector<double>, double, Vertices> createClusteringVariableSpacing(
//     mesh::PtrMesh inMesh,
//     mesh::PtrMesh outMesh,
//     double relativeOverlap,
//     unsigned int verticesPerCluster,
//     bool projectClustersToInput)
// {
//   precice::logging::Logger _log{"impl::createClusteringVariableSpacing"};
//   PRECICE_TRACE();
//   PRECICE_ASSERT(relativeOverlap < 1);
//   PRECICE_ASSERT(verticesPerCluster > 0);
//   PRECICE_ASSERT(inMesh->getDimensions() == outMesh->getDimensions());

//   // 边界条件：无顶点时直接返回空结果
//   if (inMesh->empty() || (outMesh->empty() && !outMesh->isJustInTime())) {
//     std::vector<unsigned int> emptyCounts;
//     std::vector<double> emptyRadii;
//     return {emptyCounts, emptyRadii, double{}, Vertices{}};
//   }
//   PRECICE_ASSERT(!inMesh->empty());
//   PRECICE_ASSERT(!outMesh->empty() || outMesh->isJustInTime());

//   // 配置参数：是否从边界框边缘开始生成簇（投影模式下启用，确保覆盖边缘）
//   const bool startGridAtEdge = projectClustersToInput;
//   const int inDim = inMesh->getDimensions();  // 空间维度（2D/3D）
//   PRECICE_DEBUG("Relative overlap: {}, Vertices per cluster: {}, Dimension: {}",
//                 relativeOverlap, verticesPerCluster, inDim);


//   // -------------------------- Step 1: 计算全局边界框（手动计算避免缓存错误） --------------------------
//   precice::mesh::BoundingBox localBB = inMesh->index().getRtreeBounds();
//   // 调试断言：验证边界框正确性
// #ifndef NDEBUG
//   precice::mesh::BoundingBox bbCheck(inDim);
//   for (const mesh::Vertex &vertex : inMesh->vertices()) {
//     bbCheck.expandBy(vertex);
//   }
//   PRECICE_ASSERT(bbCheck == localBB);
// #endif
//   precice::mesh::BoundingBox globalBB = localBB;  // 全局边界框（子区域划分基准）


//   // -------------------------- Step 2: 特殊情况处理：网格顶点过少时生成单个簇 --------------------------
//   if (inMesh->nVertices() < verticesPerCluster * 2) {
//     double clusterRadius = globalBB.longestEdgeLength();
//     // 单个顶点时默认半径为1（避免零半径）
//     if (clusterRadius < math::NUMERICAL_ZERO_DIFFERENCE) {
//       PRECICE_WARN("Single vertex mesh detected: setting default cluster radius to 1.0");
//       clusterRadius = 1.0;
//     }
//     Vertices singleCenter{mesh::Vertex(globalBB.center(), 0)};
//     std::vector<unsigned int> singleCount;
//     singleCount.emplace_back(1);
//     std::vector<double> singleRadius;
//     singleRadius.emplace_back(clusterRadius * 2);
//     return {singleCount, singleRadius, clusterRadius * 2, singleCenter}; // 半径设为边界框最长边2倍（确保覆盖所有顶点）
//   }


//   // -------------------------- Step 3: 调用estimateDensityField获取子区域密度信息 --------------------------
//   DensityField densityField = estimateDensityField(verticesPerCluster, inMesh, globalBB);
//   const double medianRadius = densityField.medianRadius;  // 全局中位数半径（基准）
//   PRECICE_DEBUG("Global median cluster radius: {}, Number of density samples: {}",
//                 medianRadius, densityField.sampleIDs.size());


//   // -------------------------- Step 4: 划分笛卡尔子区域（2D=4个，3D=8个） --------------------------
//   std::vector<SubRegion> subRegions;
//   const Eigen::VectorXd midPoint = (globalBB.maxCorner() + globalBB.minCorner()) / 2.0;  // 全局中心点（划分基准）

//   if (inDim == 2) {
//     // 2D子区域：按x/y中点分为4个等大区域
//     const double minX = globalBB.minCorner()(0), maxX = globalBB.maxCorner()(0);
//     const double minY = globalBB.minCorner()(1), maxY = globalBB.maxCorner()(1);
//     const double midX = midPoint(0), midY = midPoint(1);

//     // 左下区域 (minX, minY) → (midX, midY)
//     subRegions.emplace_back(SubRegion{precice::mesh::BoundingBox({minX, minY, midX, midY}), 0.0, {}});
//     // 左上区域 (minX, midY) → (midX, maxY)
//     subRegions.emplace_back(SubRegion{precice::mesh::BoundingBox({minX, midY, midX, maxY}), 0.0, {}});
//     // 右上区域 (midX, midY) → (maxX, maxY)
//     subRegions.emplace_back(SubRegion{precice::mesh::BoundingBox({midX, midY, maxX, maxY}), 0.0, {}});
//     // 右下区域 (midX, minY) → (maxX, midY)
//     subRegions.emplace_back(SubRegion{precice::mesh::BoundingBox({midX, minY, maxX, midY}), 0.0, {}});

//   } else if (inDim == 3) {
//     // 3D子区域：按x/y/z中点分为8个等大区域（所有min/mid组合）
//     const double minX = globalBB.minCorner()(0), maxX = globalBB.maxCorner()(0);
//     const double minY = globalBB.minCorner()(1), maxY = globalBB.maxCorner()(1);
//     const double minZ = globalBB.minCorner()(2), maxZ = globalBB.maxCorner()(2);
//     const double midX = midPoint(0), midY = midPoint(1), midZ = midPoint(2);

//     // 8种x/y/z组合（false=min→mid，true=mid→max）
//     const std::vector<std::array<bool, 3>> xyzFlags = {
//       {false, false, false}, {false, true, false},
//       {true, true, false},  {true, false, false},
//       {true, false, true},  {false, false, true},
//       {false, true, true},   {true, true, true}
//     };

//     for (const auto &flags : xyzFlags) {
//       const double sMinX = flags[0] ? midX : minX;
//       const double sMaxX = flags[0] ? maxX : midX;
//       const double sMinY = flags[1] ? midY : minY;
//       const double sMaxY = flags[1] ? maxY : midY;
//       const double sMinZ = flags[2] ? midZ : minZ;
//       const double sMaxZ = flags[2] ? maxZ : midZ;

//       subRegions.emplace_back(SubRegion{
//         precice::mesh::BoundingBox({sMinX, sMinY, sMinZ, sMaxX, sMaxY, sMaxZ}),
//         0.0, {}
//       });
//     }
//   }

//   // 断言：子区域数量正确（2D=4，3D=8）
//   PRECICE_ASSERT((inDim == 2 && subRegions.size() == 4) || (inDim == 3 && subRegions.size() == 8),
//                 "SubRegion count error: dim={}, count={}", inDim, subRegions.size());


//   // -------------------------- Step 5: 子区域与密度取样匹配（一一对应） --------------------------
//   for (size_t i = 0; i < densityField.sampleIDs.size(); ++i) {
//     const VertexID sampleVID = densityField.sampleIDs[i];
//     const Eigen::VectorXd &sampleCoord = inMesh->vertex(sampleVID).getCoords();
//     const double sampleRad = densityField.sampleRadii[i];

//     subRegions[i].sampleRadius = sampleRad;
//   }

//   // 断言：所有子区域均匹配到取样半径
//   for (const auto &sub : subRegions) {
//     PRECICE_ASSERT(sub.sampleRadius > math::NUMERICAL_ZERO_DIFFERENCE);
//   }


//   // -------------------------- Step 6: 计算基准间距baseDistance（全局均匀基准） --------------------------
//   // 1. 全局最大簇间距（基于中位数半径，控制重叠）
//   const double maxGlobalCenterDist = std::sqrt(4.0 / inDim) * medianRadius * (1.0 - relativeOverlap);
//   // 2. 全局簇数量（每个维度向上取整，确保覆盖全局）
//   std::array<unsigned int, 3> nClustersGlobal{1, 1, 1};
//   // 3. 基准间距（每个维度的均匀间距）
//   std::vector<double> baseDistance(inDim);
//   for (int d = 0; d < inDim; ++d) {
//     nClustersGlobal[d] = std::ceil(std::max(1.0, globalBB.getEdgeLength(d) / maxGlobalCenterDist));
//     baseDistance[d] = globalBB.getEdgeLength(d) / nClustersGlobal[d];
//   }
//   PRECICE_DEBUG("Global base distances: [{}, {}]",
//                 (inDim >= 1 ? std::to_string(baseDistance[0]) : ""),
//                 (inDim >= 2 ? std::to_string(baseDistance[1]) : "") + (inDim >= 3 ? ", " + std::to_string(baseDistance[2]) : ""));


//   // -------------------------- Step 7: 计算每个子区域的局部间距 --------------------------
//   for (auto &sub : subRegions) {
//     sub.distance.resize(inDim);
//     for (int d = 0; d < inDim; ++d) {
//       // 局部间距 = 基准间距 × (子区域取样半径 / 全局中位数半径)
//       // 密度越高（取样半径越小），局部间距越小，簇中心越密集
//       sub.distance[d] = baseDistance[d] * (sub.sampleRadius / medianRadius);
//       // 防止间距过小（避免浮点误差导致的无限簇数量）
//       sub.distance[d] = std::max(sub.distance[d], math::NUMERICAL_ZERO_DIFFERENCE * 100);
//     }
//     PRECICE_DEBUG("SubRegion distance (rad={}): [{}, {}]",
//                   sub.sampleRadius,
//                   (inDim >= 1 ? std::to_string(sub.distance[0]) : ""),
//                   (inDim >= 2 ? std::to_string(sub.distance[1]) : "") + (inDim >= 3 ? ", " + std::to_string(sub.distance[2]) : ""));
//   }


//   // -------------------------- Step 8: 在每个子区域内生成簇中心及对应半径 --------------------------
//   Vertices clusterCenters;       // 最终簇中心向量
//   std::vector<double> clusterRads;  // 各子域内簇半径向量
//   std::vector<unsigned int> clusterCounts;
//   VertexID currentRegionId = 0;

//   // 计算每个子域内簇的数量，统一存储在allSubClusterCount，并初始化总簇中心向量大小
//   size_t totalClusterCount = 0;
//   std::vector<std::array<unsigned int, 3>> allSubClusterCount;
//   for (const auto &sub : subRegions) {
//     std::array<unsigned int, 3> subClusterCount{1, 1, 1};
//     for (int d = 0; d < inDim; ++d) {
//       if (sub.distance[d] > math::NUMERICAL_ZERO_DIFFERENCE) {
//         subClusterCount[d] = std::ceil(sub.subBB.getEdgeLength(d) / sub.distance[d]);
//       }
//     }
//     unsigned int nClustersLocal = std::accumulate(subClusterCount.begin(), subClusterCount.end(), 1U, std::multiplies<unsigned int>());
//     totalClusterCount += nClustersLocal;
//     clusterCounts.push_back(nClustersLocal);
//     allSubClusterCount.push_back(subClusterCount);
//   }
//   clusterCenters.resize(totalClusterCount, mesh::Vertex({std::vector<double>(inDim), -1}));

//   for (const auto &sub : subRegions) {
//     const auto &subBB = sub.subBB;
//     const auto &subDist = sub.distance;
//     clusterRads.push_back(sub.sampleRadius);  // 对应子区域内簇半径
//     std::vector<double> centerCoords(inDim);

//     // 1. 计算子区域内每个维度的簇数量（向上取整，确保覆盖子区域）
//     std::array<unsigned int, 3> subClusterCount = allSubClusterCount[currentRegionId++];

//     // 2. 子区域内簇中心的起始坐标（边缘模式/内部模式）
//     std::vector<double> subStart(inDim);
//     for (int d = 0; d < inDim; ++d) {
//       subStart[d] = subBB.minCorner()(d);
//       if (!startGridAtEdge) {  // 非边缘模式：起始位置偏移半个间距（避免簇中心贴边）
//         subStart[d] += 0.5 * subDist[d];
//       }
//     }

//     // Print some information
//     PRECICE_DEBUG("Local cluster distribution: {}", subClusterCount);

//     // 3. 生成子区域内的簇中心（2D/3D分别处理）
//     if (inDim == 2) {
//       centerCoords[0] = subStart[0];
//       for (unsigned int x = 0; x < subClusterCount[0]; ++x, centerCoords[0] += subDist[0]) {
//         centerCoords[1] = subStart[1];
//         for (unsigned int y = 0; y < subClusterCount[1]; ++y, centerCoords[1] += subDist[1]) {
//           // 添加簇中心和对应半径
//           auto id = zCurve(std::array<unsigned int, 3>{{x, y, 0}}, subClusterCount);
//           PRECICE_ASSERT(id < static_cast<int>(clusterCenters.size()) && id >= 0);
//           clusterCenters[id] = mesh::Vertex({centerCoords, id});
//         }
//       }
//     } else if (inDim == 3) {
//       centerCoords[0] = subStart[0];
//       for (unsigned int x = 0; x < subClusterCount[0]; ++x, centerCoords[0] += subDist[0]) {
//         centerCoords[1] = subStart[1];
//         for (unsigned int y = 0; y < subClusterCount[1]; ++y, centerCoords[1] += subDist[1]) {
//           centerCoords[2] = subStart[2];
//           for (unsigned int z = 0; z < subClusterCount[2]; ++z, centerCoords[2] += subDist[2]) {
//             auto id = zCurve(std::array<unsigned int, 3>{{x, y, z}}, subClusterCount);
//             PRECICE_ASSERT(id < static_cast<int>(clusterCenters.size()) && id >= 0);
//             clusterCenters[id] = mesh::Vertex({centerCoords, id});
//           }
//         }
//       }
//     }
//   }
//   PRECICE_DEBUG("Generated {} initial cluster centers", clusterCenters.size());


//   // -------------------------- Step 9: 标记空簇（支持不同簇半径） --------------------------
//   // 自定义空簇标记逻辑：每个簇使用自身半径判断是否包含网格顶点
//   auto tagEmptyClustersVarRad = [&](Vertices &centers, const std::vector<double> &rads, mesh::PtrMesh mesh) {
//     // std::vector<std::array<unsigned int, 3>> allSubClusterCount;
//     int currentVertexIndex = 0;
//     for (size_t i = 0; i < rads.size(); ++i) {
//       auto subVertexBegin = centers.begin() + currentVertexIndex;
//       auto subVertexEnd = subVertexBegin + clusterCounts[i];
//       currentVertexIndex += clusterCounts[i];
//       std::for_each(subVertexBegin, subVertexEnd, [&](auto &v) {
//         if (!v.isTagged() && !mesh->index().isAnyVertexInsideBox(v, rads[i])) {
//           v.tag();
//           clusterCounts[i]--;
//         }
//       });
//     }
//   };

//   // 1. 基于inMesh标记空簇（必须包含输入网格顶点）
//   tagEmptyClustersVarRad(clusterCenters, clusterRads, inMesh);
//   // 2. 基于outMesh标记空簇（非即时生成网格需包含输出网格顶点）
//   if (!outMesh->isJustInTime()) {
//     tagEmptyClustersVarRad(clusterCenters, clusterRads, outMesh);
//   }
//   PRECICE_DEBUG("Non-empty clusters after tagging: {} / {}", 
//     std::count_if(clusterCenters.begin(), clusterCenters.end(), [](const auto &v) { return !v.isTagged(); }), 
//     clusterCenters.size()
//   );


//   // -------------------------- Step 10: 可选：将簇中心投影到inMesh（适配不规则网格） --------------------------
//   // if (projectClustersToInput) {
//   //   // 1. 投影簇中心到inMesh最近顶点
//   //   projectClusterCentersToinputMesh(clusterCenters, inMesh);

//   //   // 2. 计算去重阈值（取所有子区域间距的最小值，避免过密簇）
//   //   double minSubDist = std::numeric_limits<double>::max();
//   //   for (const auto &sub : subRegions) {
//   //     for (double d : sub.distance) {
//   //       minSubDist = std::min(minSubDist, d);
//   //     }
//   //   }
//   //   const double duplicateThreshold = 0.4 * minSubDist;  // 经验值：40%最小间距
//   //   PRECICE_DEBUG("Duplicate threshold: {} (min sub-region distance: {})",
//   //                 duplicateThreshold, minSubDist);

//   //   // 3. 标记重复簇（基于zCurve邻居检索）
//   //   if (inDim == 3 && nClustersGlobal[2] > 1) {  // 3D且z方向有多个簇
//   //     tagDuplicateCenters<3>(clusterCenters, nClustersGlobal, duplicateThreshold);
//   //   } else {  // 2D或3D但z方向无簇
//   //     tagDuplicateCenters<2>(clusterCenters, nClustersGlobal, duplicateThreshold);
//   //   }

//   //   // 4. 再次标记空簇（投影后位置变化可能导致空簇）
//   //   if (!outMesh->isJustInTime()) {
//   //     tagEmptyClustersVarRad(clusterCenters, clusterRads, outMesh);
//   //   }

//   //   // 调试信息：去重后剩余簇数量
//   //   PRECICE_DEBUG("Clusters after deduplication: {} / {}", 
//   //     std::count_if(clusterCenters.begin(), clusterCenters.end(), [](const auto &v) { return !v.isTagged(); }), 
//   //     clusterCenters.size());
//   // }
//   if (projectClustersToInput) {
//     PRECICE_DEBUG("projectClustersToInput not implemented: variable spacing with projection.");
//   }

//   // -------------------------- Step 11: 移除标记的无效簇（空簇/重复簇） --------------------------
//   // 同步移除簇中心和对应半径（保持一一对应）
//   removeTaggedVertices(clusterCenters);

//   PRECICE_ASSERT(std::none_of(clusterCenters.begin(), clusterCenters.end(), [](auto &v) { return v.isTagged(); }));
//   PRECICE_CHECK(clusterCenters.size() > 0, "Too many vertices have been filtered out.");

//   return {clusterCounts, clusterRads, medianRadius, clusterCenters};
// }

} // namespace precice::mapping::impl
