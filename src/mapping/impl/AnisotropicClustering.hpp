#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <unordered_map> //0128
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include "mesh/Mesh.hpp"
#include "query/Index.hpp"
#include "AnisotropicVertexCluster.hpp"
#include "precice/impl/Types.hpp"
#include "mapping/RadialBasisFctSolver.hpp"
#include "logging/Logger.hpp"
#include "logging/LogMacros.hpp"

#include "mapping/impl/CreateClustering.hpp"

namespace precice::mapping::impl {

using Vertices = std::vector<mesh::Vertex>;

namespace {

/**
 * @brief Voxel Downsampling to get pilot points
 */
std::vector<Eigen::Vector3d> samplePilotPositions(const mesh::PtrMesh inMesh)
{
    precice::mesh::BoundingBox bb = inMesh->index().getRtreeBounds();

    PRECICE_ASSERT(!bb.isDefault(), "Invalid bounding box.");
    
    const int dim = inMesh->getDimensions();
    // Use a fixed split or adaptive? 2 splits per dim = 4 or 8 points. Might be too few for global stats.
    // Let's increase to 4 splits per dim -> 64 points in 3D. Better for statistics.
    int nSplits = 4; 
    
    std::vector<double> distances(dim);
    std::vector<double> start(dim);
    for (int d = 0; d < dim; ++d) {
        distances[d] = bb.getEdgeLength(d) / nSplits;
        start[d] = bb.minCorner()[d] + distances[d] / 2.0;
    }

    std::vector<Eigen::Vector3d> pilotPositions;
    std::vector<double> centerCoords(dim);
    
    if (dim == 2) {
        for (int i = 0; i < nSplits; ++i) {
            for (int j = 0; j < nSplits; ++j) {
                centerCoords[0] = start[0] + i * distances[0];
                centerCoords[1] = start[1] + j * distances[1];
                pilotPositions.emplace_back(Eigen::Vector3d(centerCoords[0], centerCoords[1], 0.0));
            }
        }
    } else if (dim == 3) {
        for (int i = 0; i < nSplits; ++i) {
            for (int j = 0; j < nSplits; ++j) {
                for (int k = 0; k < nSplits; ++k) {
                    centerCoords[0] = start[0] + i * distances[0];
                    centerCoords[1] = start[1] + j * distances[1];
                    centerCoords[2] = start[2] + k * distances[2];
                    pilotPositions.emplace_back(Eigen::Vector3d(centerCoords[0], centerCoords[1], centerCoords[2]));
                }
            }
        }
    }
    return pilotPositions;
}
/**
 * @brief Compute global anisotropy parameters based on pilot positions in the input mesh.
 *        Uses PCA to determine the main directions and their variances.
 * 
 * @param[in] inMesh The input mesh.
 * @param[in] baseRadius The base radius for the smallest semi-axis.
 * @param[in] kNeighbors Number of neighbors to consider for local covariance.
 * 
 * @return GlobalAnisotropyParams The computed global anisotropy parameters.
 */
/**
 * @brief Compute global anisotropy parameters with Metric-based Neighborhood Search
 */
GlobalAnisotropyParams computeGlobalAnisotropyParams(
    const mesh::PtrMesh inMesh,
    double baseRadius,
    bool useDynamicRatio,
    bool autoFallback,
    double staticRatio1,
    double staticRatio2)
{
    precice::logging::Logger _log{"impl::AnisotropicClustering::computeGlobalAnisotropyParams"};

    if (staticRatio1 >= 1.0 || staticRatio2 >= 1.0) {
        PRECICE_DEBUG("GlobalAnisotropy: Using static anisotropic ratios [{}, {}].", staticRatio1, staticRatio2);
    } else {
        if (useDynamicRatio) {
            PRECICE_DEBUG("GlobalAnisotropy: Using dynamic anisotropic ratio based on local PCA analysis.");
        } else {
            PRECICE_DEBUG("GlobalAnisotropy: Using Geometry ratio.");
        }
    }

    // 获取全局均匀取样点
    std::vector<Eigen::Vector3d> pilotPositions = samplePilotPositions(inMesh);
    const int nPos = pilotPositions.size();
    
    const int dim = inMesh->getDimensions();
    Eigen::Matrix3d globalCov = Eigen::Matrix3d::Zero();
    int validPilots = 0;
    std::vector<Eigen::Vector3d> validLocalDirections;
    const double pcaSearchRadius = baseRadius * 2.0; // 基于物理尺度的搜索半径，2.0是经验值
    const size_t MIN_NEIGHBORS = 6; // 最小点数要求，防止极其稀疏区域导致 PCA 数值错误
    const double ANISOTROPY_THRESHOLD = 1.3; // 筛选阈值

    // 遍历采样点
    for(const auto& p : pilotPositions) {
        mesh::Vertex vertexPilot(p, -1); // 临时 Vertex 对象，仅用于查询
        std::vector<int> neighbors = inMesh->index().getVerticesInsideBox(vertexPilot, pcaSearchRadius);
        
        if(neighbors.size() < MIN_NEIGHBORS) continue; 
        
        // 构建数据矩阵
        Eigen::MatrixXd data(neighbors.size(), 3);
        for(size_t i = 0; i < neighbors.size(); ++i) {
            const auto& v = inMesh->vertex(neighbors[i]);
            if(v.getDimensions() == 3) 
                data.row(i) = v.getCoords();
            else 
                data.row(i) << v.coord(0), v.coord(1), 0.0;
        }
        
        // 计算协方差
        Eigen::Vector3d mean = data.colwise().mean();
        Eigen::MatrixXd centered = data.rowwise() - mean.transpose();
        // 归一化：除以 (N-1) 得到无偏估计
        Eigen::Matrix3d cov = (centered.adjoint() * centered) / double(data.rows() - 1);
        
        // --- 局部特征分析 ---
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> localSolver(cov);
        Eigen::Vector3d locEvals = localSolver.eigenvalues(); // 0:Min, 2:Max
        
        double minLambda, maxLambda;

        // --- 2D/3D 特征值选择 ---
        if (dim == 2) {
            // 2D 情况下，locEvals(0) 是 Z 轴的 0 (或极小噪声)，应忽略。
            minLambda = std::max(locEvals(1), 1e-10);
            maxLambda = std::max(locEvals(2), 1e-10);
        } else {
            // 3D 情况下，0 是最小轴
            minLambda = std::max(locEvals(0), 1e-10);
            maxLambda = std::max(locEvals(2), 1e-10);
        }
        
        // 计算几何长宽比
        double localRatio = std::sqrt(maxLambda / minLambda);
        
        // 过滤噪声/各向同性点
        if (localRatio < ANISOTROPY_THRESHOLD) {
            continue; 
        }

        // --- 加权累加 (可选) ---
        // double weight = std::log(localRatio); // 这种激进加权可以强化主方向
        globalCov += cov;
        validLocalDirections.push_back(localSolver.eigenvectors().col(2));
        validPilots++;
    }

    GlobalAnisotropyParams params;
    // 初始化默认值
    params.rotation = Eigen::Matrix3d::Identity();
    params.semiAxes = Eigen::Vector3d::Constant(baseRadius);

    // 全局统计与回退
    if(validPilots == 0) {
        if (autoFallback) {
            PRECICE_INFO("GlobalAnisotropy: No reliable anisotropic features found. Fallback to spherical.");
            params.fallbackToSpherical = true;
        } else {
            PRECICE_INFO("GlobalAnisotropy: No reliable anisotropic features found, but automatic fallback is disabled. Continuing with isotropic anisotropic-cluster parameters.");
        }
        params.coverSearchRadius = baseRadius;
        double invRad2 = 1.0 / (baseRadius * baseRadius);
        params.inverseCovariance = Eigen::Matrix3d::Identity() * invRad2;
        return params;
    }
    
    globalCov /= validPilots;
    
    // 4. 全局分解
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigSolver(globalCov);
    Eigen::Vector3d eigenvalues = eigSolver.eigenvalues(); 
    Eigen::Matrix3d eigenvectors = eigSolver.eigenvectors();
    
    eigenvalues = eigenvalues.cwiseMax(1e-10);
    
    Eigen::Matrix3d R;
    // Columns are the principal directions ordered by descending eigenvalue:
    // col(0) = major axis, col(1) = secondary axis, col(2) = minor axis.
    R.col(0) = eigenvectors.col(2);
    R.col(1) = eigenvectors.col(1);
    R.col(2) = eigenvectors.col(0);
    // --- 2D 模式下的旋转矩阵清洗 ---
    if (dim == 2) {
        // 强制 R 的结构为绕 Z 轴旋转
        // 1. 确保第 3 列 (Z轴方向) 是 (0, 0, 1)
        R(0, 2) = 0.0;
        R(1, 2) = 0.0;
        R(2, 2) = 1.0;
        
        // 2. 确保第 3 行 (Z轴分量) 是 (0, 0, 1)
        R(2, 0) = 0.0;
        R(2, 1) = 0.0;
        
        // 3. 重新归一化前两列（消除潜在的非正交误差）
        R.col(0).normalize();
        R.col(1).normalize();
    }
    params.rotation = R;

    // Coherence 计算
    double coherenceScore = 0.0;
    Eigen::Vector3d globalDir = eigenvectors.col(2); 

    for (const auto& localDir : validLocalDirections) {
        coherenceScore += std::abs(localDir.dot(globalDir)); 
    }
    coherenceScore /= validPilots;

    // 动态限制 (Dynamic Limit)
    const double minScoreThreshold = 0.35;
    const double absoluteMinRatio  = 1.5;
    const double absoluteMaxRatio  = 3.5; 

    double dynamicMaxRatio = 1.0;
    if (useDynamicRatio) {
        dynamicMaxRatio = absoluteMinRatio;
        if (coherenceScore > minScoreThreshold) {
            double t = (coherenceScore - minScoreThreshold) / (1.0 - minScoreThreshold);
            dynamicMaxRatio = absoluteMinRatio + (t * t) * (absoluteMaxRatio - absoluteMinRatio);
        }
    }
    
    Eigen::Vector3d semiAxesRatio;
    semiAxesRatio(0) = std::sqrt(eigenvalues(2));
    semiAxesRatio(1) = std::sqrt(eigenvalues(1));
    semiAxesRatio(2) = std::sqrt(eigenvalues(0));

    double geomRatio0 = 1.0;
    double geomRatio1 = 1.0;
    double finalRatio0, finalRatio1;
    double norConstant;
    if (dim == 2) {
        geomRatio0 = semiAxesRatio(0) / semiAxesRatio(1);

        if (staticRatio1 >= 1.0) {
            finalRatio0 = staticRatio1;
        } else if (useDynamicRatio) {
            finalRatio0 = std::min(geomRatio0, dynamicMaxRatio);
        } else {
            finalRatio0 = geomRatio0;
        }

        norConstant = std::sqrt(finalRatio0 * 1.0);

        params.semiAxes(0) = baseRadius * finalRatio0 / norConstant;
        params.semiAxes(1) = baseRadius * 1.0 / norConstant;
        params.semiAxes(2) = params.semiAxes(1);
    } else {
        geomRatio0 = semiAxesRatio(0) / semiAxesRatio(2);  // sqrt(λmax/λmin)
        geomRatio1 = semiAxesRatio(1) / semiAxesRatio(2);  // sqrt(λmid/λmin)

        if (useDynamicRatio) {
            finalRatio0 = std::min(geomRatio0, dynamicMaxRatio);
            finalRatio1 = std::min(geomRatio1, dynamicMaxRatio);
        } else {
            // Static/default 3D shape: the major axis is stretched, the others stay at the base scale.
            finalRatio0 = (staticRatio1 >= 1.0) ? staticRatio1 : 3.0;
            finalRatio1 = (staticRatio2 >= 1.0) ? staticRatio2 : 1.0;
        }

        finalRatio0 = std::max(finalRatio0, 1.0);
        finalRatio1 = std::max(finalRatio1, 1.0);
        finalRatio1 = std::min(finalRatio1, finalRatio0);

        PRECICE_DEBUG("Using 3D ratios [{}, {}] (geom=[{}, {}], dynamic={}, upperLimit={})",
                      finalRatio0, finalRatio1, geomRatio0, geomRatio1,
                      useDynamicRatio ? "ON" : "OFF", dynamicMaxRatio);

        norConstant = std::cbrt(finalRatio0 * finalRatio1 * 1.0);

        params.semiAxes(0) = baseRadius * finalRatio0 / norConstant;
        params.semiAxes(1) = baseRadius * finalRatio1 / norConstant;
        params.semiAxes(2) = baseRadius * 1.0 / norConstant;
    }

    const double fallbackCoherenceThreshold = 0.35;
    const double fallbackAnisotropyThreshold = 1.15;
    const bool noClearPrincipalAxis = (geomRatio0 < fallbackAnisotropyThreshold);
    if (noClearPrincipalAxis || coherenceScore < fallbackCoherenceThreshold) {
        if (autoFallback) {
            PRECICE_INFO("GlobalAnisotropy: unreliable principal-axis estimate (geomRatio=[{}, {}], coherence={}). Fallback to spherical.",
                         geomRatio0, geomRatio1, coherenceScore);
            params.fallbackToSpherical = true;
            params.rotation = Eigen::Matrix3d::Identity();
            params.semiAxes = Eigen::Vector3d::Constant(baseRadius);
            params.coverSearchRadius = baseRadius;
            double invRad2 = 1.0 / (baseRadius * baseRadius);
            params.inverseCovariance = Eigen::Matrix3d::Identity() * invRad2;
            return params;
        } else {
            PRECICE_INFO("GlobalAnisotropy: unreliable principal-axis estimate (geomRatio=[{}, {}], coherence={}), but automatic fallback is disabled.",
                         geomRatio0, geomRatio1, coherenceScore);
        }
    }
    params.coverSearchRadius = params.semiAxes.maxCoeff();

    PRECICE_DEBUG("AnisotropyParams: pilots={}/{}, score={}, allowedRatio={}", validPilots, nPos, coherenceScore, dynamicMaxRatio);
    PRECICE_DEBUG("GeomRatio: [{}, {}] -> FinalRatio: [{}, {}]", geomRatio0, geomRatio1, finalRatio0, finalRatio1);
    PRECICE_DEBUG("FallbackToSpherical: {}", params.fallbackToSpherical ? "YES" : "NO");
    PRECICE_DEBUG("Rotation Matrix R:\n[{}, {}, {}]\n[{}, {}, {}]\n[{}, {}, {}]",
                  R(0,0), R(0,1), R(0,2),
                  R(1,0), R(1,1), R(1,2),
                  R(2,0), R(2,1), R(2,2));

    // 逆协方差 M
    Eigen::Matrix3d S_inv2 = Eigen::Matrix3d::Zero();
    S_inv2(0, 0) = 1.0 / (params.semiAxes(0) * params.semiAxes(0));
    S_inv2(1, 1) = 1.0 / (params.semiAxes(1) * params.semiAxes(1));
    S_inv2(2, 2) = 1.0 / (params.semiAxes(2) * params.semiAxes(2));
    params.inverseCovariance = R * S_inv2 * R.transpose();

    return params;
}

/**
 * @brief Generate anisotropic cluster centers in the ellipsoid principal-axis frame.
 *        The centers are placed on a Cartesian lattice in normalized coordinates,
 *        which gives a coverage guarantee for overlap = 0 and preserves that guarantee
 *        for any denser spacing induced by overlap > 0 or by ceil-based discretization.
 * 
 * @param[in] globalBB The global bounding box for cluster center generation.
 * @param[in] params Global anisotropy parameters.
 * @param[in] overlap The overlap of the local cluster.
 * @param[in] dim Spatial dimension.
 * @param[in] projectToInput Whether a later projection-to-input step is expected.
 * @param[out] actualSpacing Optional actual lattice spacing in the principal-axis frame.
 * @return Vertices The generated cluster centers in global coordinates.
 */
Vertices createClusterCenters(
    const precice::mesh::BoundingBox& globalBB,
    const GlobalAnisotropyParams& params,
    double overlap,
    int dim,
    bool projectToInput,
    Eigen::Vector3d *actualSpacing = nullptr)
{
    PRECICE_ASSERT(overlap < 1.0);

    // Define transformation.
    Eigen::Vector3d bbCenter = globalBB.center();
    Eigen::Matrix3d T_inv = params.rotation.transpose();
    const Eigen::Vector3d& radii = params.semiAxes;
    
    // Calculate local bounds in the principal-axis frame.
    Eigen::Vector3d local_min = Eigen::Vector3d::Constant(std::numeric_limits<double>::max());
    Eigen::Vector3d local_max = Eigen::Vector3d::Constant(std::numeric_limits<double>::lowest());
        
    for(int i=0; i<8; ++i) {
        Eigen::Vector3d p = Eigen::Vector3d::Zero();
        
        if (dim == 2 && i >= 4) break;

        p[0] = (i & 1) ? globalBB.maxCorner()[0] : globalBB.minCorner()[0];
        p[1] = (i & 2) ? globalBB.maxCorner()[1] : globalBB.minCorner()[1];
        if (dim == 3) {
            p[2] = (i & 4) ? globalBB.maxCorner()[2] : globalBB.minCorner()[2];
        }
        
        Eigen::Vector3d p_local = T_inv * (p - bbCenter);
        local_min = local_min.cwiseMin(p_local);
        local_max = local_max.cwiseMax(p_local);
    }

    Eigen::Vector3d effectiveRadii = radii;
    if (dim == 2) {
        effectiveRadii.z() = 1.0;
        local_min.z() = 0.0;
        local_max.z() = 0.0;
    }

    // In normalized coordinates y_i = x_i / r_i, the ellipsoid becomes a unit sphere.
    // A Cartesian lattice with spacing 2 / sqrt(dim) is the just-touching coverage case.
    const double normalizedSpacing = (2.0 / std::sqrt(static_cast<double>(dim))) * (1.0 - overlap);
    Eigen::Vector3d maxStep = normalizedSpacing * effectiveRadii;

    if (maxStep.minCoeff() <= 1e-9) {
        return Vertices{mesh::Vertex({globalBB.center(), 0})};
    }

    std::array<unsigned int, 3> nClusters{1, 1, 1};
    Eigen::Vector3d distances = Eigen::Vector3d::Zero();
    Eigen::Vector3d start = local_min;

    for (int d = 0; d < dim; ++d) {
        const double edgeLength = local_max[d] - local_min[d];
        if (edgeLength <= math::NUMERICAL_ZERO_DIFFERENCE) {
            nClusters[d] = 1;
            distances[d] = 0.0;
            continue;
        }

        nClusters[d] = static_cast<unsigned int>(std::ceil(std::max(1.0, edgeLength / maxStep[d])));
        distances[d] = edgeLength / static_cast<double>(nClusters[d]);

        if (projectToInput) {
            nClusters[d] += 1;
        } else {
            start[d] += 0.5 * distances[d];
        }
    }

    if (actualSpacing != nullptr) {
        *actualSpacing = distances;
    }

    std::vector<Eigen::Vector3d> localPosition;
    localPosition.reserve(static_cast<std::size_t>(nClusters[0]) * static_cast<std::size_t>(nClusters[1]) * static_cast<std::size_t>(nClusters[2]));

    for (unsigned int k = 0; k < nClusters[2]; ++k) {
        const double z = (dim == 3) ? (start.z() + static_cast<double>(k) * distances.z()) : 0.0;
        for (unsigned int j = 0; j < nClusters[1]; ++j) {
            const double y = start.y() + static_cast<double>(j) * distances.y();
            for (unsigned int i = 0; i < nClusters[0]; ++i) {
                const double x = start.x() + static_cast<double>(i) * distances.x();
                localPosition.emplace_back(x, y, z);
            }
        }
    }

    std::vector<Eigen::Vector3d> globalCandidates;
    globalCandidates.reserve(localPosition.size());
    for(const auto& loc : localPosition) {
        globalCandidates.push_back((params.rotation * loc) + bbCenter);
    }

    Vertices centers;
    for (const auto& pos : globalCandidates) {
        if (dim == 2) {
            centers.emplace_back(pos.head<2>(), -1);
        } else {
            centers.emplace_back(pos, -1);
        }
    }
    return centers;
}

/**
 * @brief Project generated centers to the closest mesh vertices and remove duplicates.
 * This effectively "snaps" the lattice to the manifold surface.
 */
std::vector<Eigen::Vector3d> projectAndUniqueCenters(
    const std::vector<Eigen::Vector3d>& rawCenters,
    const mesh::PtrMesh inMesh)
{
    if (rawCenters.empty()) return {};

    // 用于存储去重后的结果
    // Key: Mesh Vertex ID (int), Value: Coordinates (Vector3d)
    // 使用 std::map 会自动根据 ID 排序，也可以用 unordered_map 但我们需要确定性
    std::map<int, Eigen::Vector3d> uniqueMap;

    for (const auto& centerPos : rawCenters) {
        // 1. 投影：找到最近的网格顶点
        // getClosestVertex 返回的是 {Vertex, distance} 或包含 index 的结构，视 precice 版本而定
        // 假设 precice::query::Index::getClosestVertex 返回的是包含 .index (ID) 的对象
        auto searchResult = inMesh->index().getClosestVertex(centerPos);
        int closestID = searchResult.index; 
        
        // 2. 去重逻辑：
        // 如果这个网格顶点还没被作为簇中心，就添加进去。避免多个簇中心投影到同一个网格点
        if (uniqueMap.find(closestID) == uniqueMap.end()) {
             uniqueMap[closestID] = inMesh->vertex(closestID).getCoords();
        }
    }

    // 3. 转回 Vector 输出
    std::vector<Eigen::Vector3d> finalCenters;
    finalCenters.reserve(uniqueMap.size());
    for (const auto& kv : uniqueMap) {
        finalCenters.push_back(kv.second);
    }
    
    return finalCenters;
}

bool isCovering(const mesh::Vertex &v1, const mesh::Vertex &v2, Eigen::Matrix3d inverseCovariance)
{
    PRECICE_ASSERT(v1.getDimensions() == v2.getDimensions());
    const unsigned int dim = v1.getDimensions();

    Eigen::Vector3d pos1;
    Eigen::Vector3d pos2;

    pos1(0) = v1.coord(0);
    pos1(1) = v1.coord(1);
    pos1(2) = dim == 3 ? v1.coord(2) : 0.0;

    pos2(0) = v2.coord(0);
    pos2(1) = v2.coord(1);
    pos2(2) = dim == 3 ? v2.coord(2) : 0.0;

    Eigen::Vector3d diff = pos1 - pos2;
    return (diff.transpose() * inverseCovariance * diff) < 1 - math::NUMERICAL_ZERO_DIFFERENCE;
}

/**
 * @brief Equivalent to "tagEmptyClusters" but for Ellipsoids.
 */
void tagEmptyAnisotropicClusters(
    Vertices &clusterCenters, 
    GlobalAnisotropyParams &params,
    mesh::PtrMesh mesh)
{
    const Eigen::Matrix3d inverseCovariance = params.inverseCovariance;
    const double coverSearchRadius = params.coverSearchRadius;

    std::for_each(clusterCenters.begin(), clusterCenters.end(), [&](auto &v) {
        if (!v.isTagged()) {
            auto ids = mesh->index().getVerticesInsideBox(v, coverSearchRadius);
            if (ids.size() == 0){
                v.tag();
            }
            else {
                bool empty = true;
                for (auto id : ids) {
                    if (isCovering(v, mesh->vertex(id), inverseCovariance)) {
                        empty = false;
                        break;
                    }
                }
                if (empty == true) {
                    v.tag();
                }
            }
        }
    });
}

struct CellKey {
    int x;
    int y;
    int z;
};

struct CellKeyHash {
    std::size_t operator()(const CellKey &key) const
    {
        std::size_t h1 = std::hash<int>{}(key.x);
        std::size_t h2 = std::hash<int>{}(key.y);
        std::size_t h3 = std::hash<int>{}(key.z);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

inline bool operator==(const CellKey &lhs, const CellKey &rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

void tagDuplicateProjectedCenters(Vertices &clusterCenters, double threshold, int dim)
{
    PRECICE_ASSERT(threshold >= 0);
    if (clusterCenters.empty() || threshold <= 0.0) {
        return;
    }

    const double inverseCellSize = 1.0 / threshold;
    const double thresholdSquared = threshold * threshold;
    std::unordered_map<CellKey, std::vector<VertexID>, CellKeyHash> grid;

    auto cellKeyForVertex = [&](const mesh::Vertex &v) {
        int x = static_cast<int>(std::floor(v.coord(0) * inverseCellSize));
        int y = static_cast<int>(std::floor(v.coord(1) * inverseCellSize));
        int z = 0;
        if (dim == 3) {
            z = static_cast<int>(std::floor(v.coord(2) * inverseCellSize));
        }
        return CellKey{x, y, z};
    };

    for (VertexID idx = 0; idx < static_cast<VertexID>(clusterCenters.size()); ++idx) {
        auto &center = clusterCenters[idx];
        if (center.isTagged()) {
            continue;
        }

        const auto key = cellKeyForVertex(center);
        bool isDuplicate = false;
        for (int dz = (dim == 3 ? -1 : 0); dz <= (dim == 3 ? 1 : 0); ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const CellKey neighborKey{key.x + dx, key.y + dy, key.z + dz};
                    auto it = grid.find(neighborKey);
                    if (it == grid.end()) {
                        continue;
                    }
                    for (auto neighborId : it->second) {
                        auto &neighbor = clusterCenters[neighborId];
                        if (neighbor.isTagged()) {
                            continue;
                        }
                        const double dxVal = center.coord(0) - neighbor.coord(0);
                        const double dyVal = center.coord(1) - neighbor.coord(1);
                        const double dzVal = dim == 3 ? (center.coord(2) - neighbor.coord(2)) : 0.0;
                        const double distanceSquared = dxVal * dxVal + dyVal * dyVal + dzVal * dzVal;
                        if (distanceSquared < thresholdSquared) {
                            center.tag();
                            isDuplicate = true;
                            break;
                        }
                    }
                    if (isDuplicate) {
                        break;
                    }
                }
                if (isDuplicate) {
                    break;
                }
            }
            if (isDuplicate) {
                break;
            }
        }

        if (!center.isTagged()) {
            grid[key].push_back(idx);
        }
    }
}

} // namespace

/**
 * @brief Generate global anisotropic params and create Anisotropic Clustering with Global Params
 */
inline std::tuple<GlobalAnisotropyParams, Vertices> createAnisotropicClustering(
    const mesh::PtrMesh inMesh,
    const mesh::PtrMesh outMesh, 
    unsigned int targetVerticesPerCluster,
    double overlapRatio,
    bool projectToInput,
    bool useDynamicRatio,
    bool autoFallback,
    double staticRatio1,
    double staticRatio2) 
{
    precice::logging::Logger _log{"impl::createAnisotropicClustering"};
    const int dim = inMesh->getDimensions();

    // Early exit for empty mesh - must check BEFORE getRtreeBounds()
    if(inMesh->vertices().size() == 0) {
        GlobalAnisotropyParams params;
        params.rotation = Eigen::Matrix3d::Identity();
        params.semiAxes = Eigen::Vector3d::Constant(1.0);
        params.inverseCovariance = Eigen::Matrix3d::Identity();
        params.coverSearchRadius = 1.0;
        return {params, Vertices{mesh::Vertex(Eigen::VectorXd::Zero(dim), 0)}};
    }

    // 1. Estimate Base Radius
    precice::mesh::BoundingBox globalBB = inMesh->index().getRtreeBounds();
    double baseRadius = estimateClusterRadius(targetVerticesPerCluster, inMesh, globalBB);

    // 2. Compute Global Anisotropy Params
    GlobalAnisotropyParams params = computeGlobalAnisotropyParams(inMesh, baseRadius, useDynamicRatio, autoFallback, staticRatio1, staticRatio2);

    // 3. Generate Cluster Centers
    
    Eigen::Vector3d actualSpacing = Eigen::Vector3d::Zero();
    Vertices centers = createClusterCenters(globalBB, params, overlapRatio, dim, projectToInput, &actualSpacing);

    tagEmptyAnisotropicClusters(centers, params, inMesh);
    if (!outMesh->isJustInTime()) {
        tagEmptyAnisotropicClusters(centers, params, outMesh);
    }

    if (projectToInput) {
        projectClusterCentersToinputMesh(centers, inMesh);
        const double minStep = (dim == 2) ? std::min(actualSpacing.x(), actualSpacing.y()) : actualSpacing.head(dim).minCoeff();
        const double duplicateThreshold = 0.4 * minStep;
        tagDuplicateProjectedCenters(centers, duplicateThreshold, dim);
        if (!outMesh->isJustInTime()) {
            tagEmptyAnisotropicClusters(centers, params, outMesh);
        }
    }
    removeTaggedVertices(centers);
    // Vertices centers = filterEmptyAnisotropicClusters(globalCandidates, inMesh, params);
    PRECICE_DEBUG("Generated {} centers (Anisotropic)", centers.size());

    return {params, centers};
}

} // namespace precice::mapping::impl
