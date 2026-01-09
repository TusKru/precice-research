#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include "mesh/Mesh.hpp"
#include "query/Index.hpp"
#include "AnisotropicVertexCluster.hpp"
#include "precice/impl/Types.hpp"
#include "mapping/RadialBasisFctSolver.hpp"

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
    double baseRadius)
{
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
            minLambda = std::max(locEvals(1), 1e-12);
            maxLambda = std::max(locEvals(2), 1e-12);
        } else {
            // 3D 情况下，0 是最小轴
            minLambda = std::max(locEvals(0), 1e-12);
            maxLambda = std::max(locEvals(2), 1e-12);
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
        printf("[Info] GlobalAnisotropy: No anisotropic features found. Fallback to isotropic.\n");
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
    
    // 排序: Lambda0 >= Lambda1 >= Lambda2
    Eigen::Vector3d sortedEvals;
    sortedEvals << eigenvalues(2), eigenvalues(1), eigenvalues(0);
    
    Eigen::Matrix3d R;
    R << eigenvectors.col(2), eigenvectors.col(1), eigenvectors.col(0);
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
    const double minScoreThreshold = 0.4; 
    const double absoluteMaxRatio  = 2.0; 

    double allowedMaxRatio = 1.0;
    if (coherenceScore > minScoreThreshold) {
        double t = (coherenceScore - minScoreThreshold) / (1.0 - minScoreThreshold);
        allowedMaxRatio = 1.0 + (t * t) * (absoluteMaxRatio - 1.0);
    }

    // 计算最终轴长
    double min_sqrt_lambda;
    
    if (dim == 2) {
        // 2D 下，使用 sortedEvals(1) (平面内短轴) 作为基准。
        min_sqrt_lambda = std::sqrt(std::max(sortedEvals(1), 1e-12));
    } else {
        // 3D 下，使用 sortedEvals(2) (最小轴)
        min_sqrt_lambda = std::sqrt(std::max(sortedEvals(2), 1e-12));
    }
    
    double geomRatio0 = std::sqrt(sortedEvals(0)) / min_sqrt_lambda;
    double geomRatio1 = std::sqrt(sortedEvals(1)) / min_sqrt_lambda;
    
    // 应用 Coherence 限制
    double finalRatio0 = std::min(geomRatio0, allowedMaxRatio);
    double finalRatio1 = std::min(geomRatio1, allowedMaxRatio);
    
    // 保证没有任何轴小于 baseRadius (避免空洞)
    // 策略：最短轴 = baseRadius，其他轴放大
    params.semiAxes(0) = baseRadius * finalRatio0;
    params.semiAxes(1) = baseRadius * finalRatio1;
    params.semiAxes(2) = baseRadius;
    params.coverSearchRadius = params.semiAxes.maxCoeff();
    
    printf("@@@@@@ GlobalAnisotropyParams @@@@@@\n");
    printf("Pilots: %d/%d; Score: %.3f; AllowedRatio: %.3f\n", validPilots, nPos, coherenceScore, allowedMaxRatio);
    printf("GeomRatio: [%.2f, %.2f] -> FinalRatio: [%.2f, %.2f]\n", 
           geomRatio0, geomRatio1, finalRatio0, finalRatio1);
    printf("SemiAxes: [%.4f, %.4f, %.4f]\n", 
           params.semiAxes(0), params.semiAxes(1), params.semiAxes(2));
    printf("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n");

    // 逆协方差 M
    Eigen::Matrix3d S_inv2 = Eigen::Matrix3d::Zero();
    S_inv2(0, 0) = 1.0 / (params.semiAxes(0) * params.semiAxes(0));
    S_inv2(1, 1) = 1.0 / (params.semiAxes(1) * params.semiAxes(1));
    S_inv2(2, 2) = 1.0 / (params.semiAxes(2) * params.semiAxes(2));
    params.inverseCovariance = R * S_inv2 * R.transpose();

    return params;
}

/**
 * @brief Generate a local anisotropic cluster centers within the given bounds with the given radii and overlap.
 *        Cluster centers are generated in a Hexagonal Close Packing like structure for better packing.
 * 
 * @param[in] local_bounds The local bounding box (min, max) for cluster center generation.
 * @param[in] radii The base radii of the local cluster.
 * @param[in] overlap The overlap of the local cluster.
 * @return std::vector<Eigen::Vector3d> The localPosition of the local lattice.
 */
Vertices createClusterCenters(
    const precice::mesh::BoundingBox& globalBB,
    const GlobalAnisotropyParams& params,
    double overlap,
    int dim)
{
    // Define Transformation
    Eigen::Vector3d bbCenter = globalBB.center();
    Eigen::Matrix3d T_inv = params.rotation.transpose();
    const Eigen::Vector3d& radii = params.semiAxes;
    
    // Calculate Local Generation Bounds
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
    
    local_min -= params.semiAxes;
    local_max += params.semiAxes;

    const std::pair<Eigen::Vector3d, Eigen::Vector3d>& local_bounds = {local_min, local_max};

    std::vector<Eigen::Vector3d> localPosition;

    // 理论上：
    // 2D 六边形密铺要无缝覆盖，间距(Step)最大为 sqrt(3)*R ≈ 1.732*R
    // 3D FCC密铺要无缝覆盖，间距(Step)最大为 sqrt(2)*R ≈ 1.414*R
    // 为了保证 overlap=0 时完全覆盖，使用几何覆盖因数。
    double coverage_factor = (dim == 3) ? std::sqrt(2.0) : std::sqrt(3.0);
    
    Eigen::Vector3d effective_radii = radii;

    if (dim == 2) {
        effective_radii.z() = 1.0; // 虚拟值，防止计算 step 出错
    }
    
    // 当 overlap=0, Step = coverage_factor * R (紧密咬合以覆盖空隙)
    Eigen::Vector3d step = coverage_factor * effective_radii * (1.0 - overlap);

    if(step.minCoeff() <= 1e-9) return Vertices{mesh::Vertex({globalBB.center(), 0})}; // 安全检查

    const auto& min = local_bounds.first;
    const auto& max = local_bounds.second;

    // 优化边界处理：向外扩展半个步长或一个半径，确保边界被覆盖
    Eigen::Vector3d start_pos = min - 0.5 * step;
    Eigen::Vector3d end_pos   = max + 0.5 * step;
    
    if (dim == 2) {
        start_pos.z() = min.z(); // 通常 2D 模拟在一个平面上，保持 min.z
        end_pos.z()   = min.z(); 
    }

    // 计算网格数量
    int nz = 1;
    if (dim == 3) {
        nz = static_cast<int>(std::floor((end_pos.z() - start_pos.z()) / step.z())) + 2;
    }
    int ny = static_cast<int>(std::floor((end_pos.y() - start_pos.y()) / step.y())) + 2;
    int nx = static_cast<int>(std::floor((end_pos.x() - start_pos.x()) / step.x())) + 2;

    for (int k = 0; k < nz; ++k) {
        double z = start_pos.z();
        if (dim == 3) z += k * step.z();

        // 3D 交错逻辑 (FCC/HCP layer shift)
        // 偶数层不偏，奇数层偏 (0.5, 0.5)
        double layer_offset_y = 0.0;
        double layer_offset_x = 0.0;
        
        if (dim == 3 && (k % 2 != 0)) {
             layer_offset_y = step.y() * 0.5;
             layer_offset_x = step.x() * 0.5;
        }

        for (int j = 0; j < ny; ++j) {
            double y = start_pos.y() + j * step.y() + layer_offset_y;

            // 2D/3D 行交错逻辑 (Hexagonal row shift)
            // 偶数行不偏，奇数行偏 0.5
            double row_offset_x = (j % 2 != 0) ? step.x() * 0.5 : 0.0;
            double total_offset_x = layer_offset_x + row_offset_x;
            
            // 归一化 offset，防止偏移过度
            while(total_offset_x >= step.x()) total_offset_x -= step.x();

            for (int i = 0; i < nx; ++i) {
                double x = start_pos.x() + i * step.x() + total_offset_x;
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

bool isCovering(Eigen::Vector3d &c, const mesh::Vertex &v, Eigen::Matrix3d inverseCovariance)
{
    Eigen::Vector3d vPos;
    if (v.getDimensions() == 3) {
        vPos = v.getCoords();
    } else {
        vPos << v.coord(0), v.coord(1), 0.0;
    }

    Eigen::Vector3d diff = vPos - c;
    return (diff.transpose() * inverseCovariance * diff) < 1 - math::NUMERICAL_ZERO_DIFFERENCE;
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

/**
 * @brief Filter out clusters that do not cover enough vertices based on Anisotropic metric.
 * Equivalent to "tagEmptyClusters" but for Ellipsoids.
 */
Vertices filterEmptyAnisotropicClusters(
    const std::vector<Eigen::Vector3d>& candidatePositions,
    const mesh::PtrMesh inMesh,
    const GlobalAnisotropyParams& params)
{
    Vertices validCenters;
    int currentID = 0;
    const int dim = inMesh->getDimensions();

    for (const auto& pos : candidatePositions) {
        mesh::Vertex centerV(pos, -1);
        
        // 1. 粗筛选：先用 R-Tree 找覆盖半径内的点
        auto ids = inMesh->index().getVerticesInsideBox(centerV, params.coverSearchRadius);
        
        // 2. 精筛选：使用马哈拉诺比斯距离判断椭球覆盖
        bool isNonEmpty = false;
        for (auto id : ids) {
            if (isCovering(const_cast<Eigen::Vector3d&>(pos), inMesh->vertex(id), params.inverseCovariance)) {
                isNonEmpty = true;
                break; // 只要覆盖到一个点，这个簇就是有效的
            }
        }

        if (isNonEmpty) {
            int dim = inMesh->getDimensions();
            if (dim == 2) {
                validCenters.emplace_back(pos.head<2>(), currentID++);
            } else {
                validCenters.emplace_back(pos, currentID++);
            }
        }
    }
    return validCenters;
}

} // namespace

/**
 * @brief Generate global anisotropic params and create Anisotropic Clustering with Global Params
 */
inline std::tuple<GlobalAnisotropyParams, Vertices> createAnisotropicClustering(
    const mesh::PtrMesh inMesh,
    const mesh::PtrMesh outMesh, 
    unsigned int targetVerticesPerCluster,
    double overlapRatio = 0.15,
    bool projectToInput = true) 
{
    precice::logging::Logger _log{"impl::createAnisotropicClustering"};
    // 1. Estimate Base Radius
    precice::mesh::BoundingBox globalBB = inMesh->index().getRtreeBounds();
    const int dim = inMesh->getDimensions();
    double baseRadius = estimateClusterRadius(targetVerticesPerCluster, inMesh, globalBB);
    
    // 2. Compute Global Anisotropy Params
    precice::profiling::Event e1("clustering.computeGlobalAnisotropyParams");
    GlobalAnisotropyParams params = computeGlobalAnisotropyParams(inMesh, baseRadius);
    e1.stop();

    // 3. Generate Cluster Centers
    precice::profiling::Event e2("clustering.createClusterCenters");
    if(inMesh->vertices().size() == 0) 
        return {params, Vertices{mesh::Vertex(Eigen::VectorXd::Zero(dim), 0)}};
    
    Vertices centers = createClusterCenters(globalBB, params, overlapRatio, dim);
    
    tagEmptyClusters(centers, params.semiAxes.minCoeff(), inMesh);

    if (projectToInput) {
        // globalCandidates = projectAndUniqueCenters(globalCandidates, inMesh);
        projectClusterCentersToinputMesh(centers, inMesh);
    }
    removeTaggedVertices(centers);
    // Vertices centers = filterEmptyAnisotropicClusters(globalCandidates, inMesh, params);
    e2.stop();
    printf("Generated %lu centers (Anisotropic)\n", centers.size());
    
    return {params, centers};
}

} // namespace precice::mapping::impl
