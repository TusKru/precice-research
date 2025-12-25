#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

#include "mesh/Mesh.hpp"
#include "query/Index.hpp"
#include "AnisotropicVertexCluster.hpp"
#include "precice/impl/Types.hpp"
#include "mapping/RadialBasisFctSolver.hpp"

namespace precice::mapping {

namespace {

/**
 * @brief Voxel Downsampling(2D: split in 4 parts; 3D: split in 8 parts)
 * 
 * @param[in] inMesh input mesh(source mesh) for creating guide points
 * 
 * @return a collection of pilot points(mesh::Vertex) using for geometric feature computation
*/
Vertices samplePilotPositions(const mesh::PtrMesh inMesh)
{
    precice::mesh::BoundingBox bb = inMesh->index().getRtreeBounds();

    PRECICE_ASSERT(!bb.isDefault(), "Invalid bounding box.");
    const auto bbCenter = bb.center();

    const int dim = inMesh->getDimensions();
    PRECICE_ASSERT(dim == 2 || dim == 3, "Dimension {} not supported for pilot point sampling.", dim);

    // Step 1: create voxel grid        
    int nSplits = 2; // split each dimension in 2 parts
    std::vector<double> distances(dim);
    std::vector<double> start(dim);
    for (int d = 0; d < dim; ++d) {
        distances[d] = bb.getEdgeLength(d) / nSplits;
        start[d] = bb.minCorner()[d] + distances[d] / 2.0;
    }

    Vertices pilotPositions;
    std::vector<double> centerCoords(dim);
    // Fill the centers
    if (dim == 2) {
        for (int i = 0; i < nSplits; ++i) {
            for (int j = 0; j < nSplits; ++j) {
                centerCoords[0] = start[0] + i * distances[0];
                centerCoords[1] = start[1] + j * distances[1];
                pilotPositions.emplace_back(mesh::Vertex({centerCoords, static_cast<VertexID>(pilotPositions.size())}));
            }
        }
    } else if (dim == 3) {
        for (int i = 0; i < nSplits; ++i) {
            for (int j = 0; j < nSplits; ++j) {
                for (int k = 0; k < nSplits; ++k) {
                    centerCoords[0] = start[0] + i * distances[0];
                    centerCoords[1] = start[1] + j * distances[1];
                    centerCoords[2] = start[2] + k * distances[2];
                    pilotPositions.emplace_back(mesh::Vertex({centerCoords, static_cast<VertexID>(pilotPositions.size())}));
                }
            }
        }
    }
    return pilotPositions;
}

/**
 * @brief 自适应各向异性映射的网格局部特征识别 (研究点一·步骤一)
 * 
 * @param[in] inMesh the mesh we search for the k-nearest vertices around each pilot point
 * 
 * @return a collection of LocalFeature structs representing the geometric features
 */
std::vector<PilotPoint> computePilotPoints(const mesh::PtrMesh inMesh)
{
    std::vector<PilotPoint> pilotPoints;
    const int kNearest = 20; // number of nearest neighbors to consider for PCA

    Vertices pilotPositions = samplePilotPositions(inMesh);

    for (const auto &pilotPosition : pilotPositions) {
        // Step 1: find k-nearest neighbors
        auto neighborIDs = inMesh->index().getClosestVertices(pilotPosition.getCoords(), kNearest);
        PRECICE_ASSERT(!neighborIDs.empty(), "No neighbors found for pilot point ID {}.", pilotPosition.getID());

        // Step 2: assemble data matrix
        Eigen::MatrixXd data(neighborIDs.size(), inMesh->getDimensions());
        for (size_t i = 0; i < neighborIDs.size(); ++i) {
            auto coords = inMesh->vertex(neighborIDs[i]).getCoords();
            for (int d = 0; d < inMesh->getDimensions(); ++d) {
                data(i, d) = coords[d];
            }
        }

        // Step 3: perform PCA
        Eigen::VectorXd mean = data.colwise().mean();
        Eigen::MatrixXd centered = data.rowwise() - mean.transpose();
        Eigen::MatrixXd cov = (centered.adjoint() * centered) / double(data.rows() - 1);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigSolver(cov);
        PRECICE_ASSERT(eigSolver.info() == Eigen::Success, "PCA Eigen decomposition failed for pilot point ID {}.", pilotPosition.getID());
        Eigen::VectorXd eigenvalues = eigSolver.eigenvalues().reverse();
        Eigen::MatrixXd eigenvectors = eigSolver.eigenvectors().rowwise().reverse();

        // Step 4: determine feature type
        FeatureType featureType;
        if (inMesh->getDimensions() == 3) {
            if (eigenvalues(1) / eigenvalues(0) < 0.1 && eigenvalues(2) / eigenvalues(0) < 0.1) {
                featureType = FeatureType::LINEAR;
            } else if (eigenvalues(2) / eigenvalues(1) < 0.1) {
                featureType = FeatureType::SURFACE;
            } else {
                featureType = FeatureType::VOLUMETRIC;
            }
        } else { // 2D case
            if (eigenvalues(1) / eigenvalues(0) < 0.1) {
                featureType = FeatureType::LINEAR;
            } else {
                featureType = FeatureType::VOLUMETRIC;
            }
        }

        // Step 5: store the feature
        PilotPoint pilotPoint;
        pilotPoint.position = mean.head<3>();
        pilotPoint.eigenvalues = eigenvalues.head<3>();
        pilotPoint.eigenvectors = eigenvectors.leftCols(3);
        pilotPoint.type = featureType;
        pilotPoints.push_back(pilotPoint);
    }
    return pilotPoints;
}
} // namespace

/**
 * @brief 生成各向异性簇 (研究点一·步骤二)
 * 
 * @param[in] inputMesh 待覆盖的网格 (Source Mesh)
 * @param[in] pilotPoints 第一步生成的导引点集
 * @param[in] targetVerticesPerCluster 目标顶点数/cluster (用于计算基础半径)
 * 
 * @return std::vector<AnisotropicVertexCluster> 生成的簇列表
 */
template <typename RADIAL_BASIS_FUNCTION_T>
std::tuple<std::vector<AnisotropicVertexCluster<RADIAL_BASIS_FUNCTION_T>>, double> createAnisotropicClustering(const mesh::PtrMesh inMesh,const mesh::PtrMesh outMesh,  RADIAL_BASIS_FUNCTION_T function, Polynomial polynomial, unsigned int targetVerticesPerCluster) 
{
    unsigned int mcSampleCount = 10;
    double shapeParameter = 3.0;

    // 1. 准备数据结构
    std::vector<AnisotropicVertexCluster<RADIAL_BASIS_FUNCTION_T>> clusters;
    int vertexCount = inMesh->vertices().size();
    if (vertexCount == 0) return {clusters, 0.0};

    // vector<bool>标记覆盖状态
    std::vector<bool> isCovered(vertexCount, false);
    int coveredCount = 0;

    std::vector<PilotPoint> pilotPoints = computePilotPoints(inMesh);

    // Pilot Points的查询索引，用于快速查找最近导引点
    mesh::Mesh pilotMesh("PilotMesh", 3, false);
    for(const auto& p : pilotPoints) {
        pilotMesh.createVertex(p.position);
    }
    query::Index pilotIndex(pilotMesh);

    // 输入网格的查询索引，用于计算基础半径
    query::Index inputIndex(inMesh); 

    // 随机数生成器，固定种子便于复现
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, vertexCount - 1);
    int candidateID = dist(rng);

    // 一个简单的中心点列表，用于存储生成的Cluster中心
    std::vector<Eigen::Vector3d> clusterCenters;

    // 2. 迭代生成Cluster
    // 终止条件可设为覆盖率达99%或连续多次无法找到有效新簇
    int maxFailures = 100;
    int failures = 0;

    double maxRange = 0.0;

    while (coveredCount < vertexCount) {
        // --- 步骤 A: 蒙特卡洛优化的 FPS 采样 (注意事项1) ---
        int bestCandidateID = -1;
        double maxDistToExisting = -1.0;

        // 抽取 M 个随机样本
        for (unsigned int i = 0; i < mcSampleCount; ++i) {
            int candidateID = dist(rng);
            
            // 如果已经被覆盖，跳过 (或寻找下一个未覆盖的)，可能多次无法找到簇
            if (isCovered[candidateID]) continue;

            const auto& candidatePos = inMesh->vertices()[candidateID].getCoords();

            // 计算该候选点到"现有簇"的最短距离
            double distToClosestCluster = std::numeric_limits<double>::max();
            
            if (!clusterCenters.empty()) {
                // 这里简化为线性扫描现有中心。
                // 随着簇数量增加，这里可以用 query::Index 优化，但考虑到FPS只需采样M个，
                // M * N_clusters 的开销通常可接受。
                for (const auto& center : clusterCenters) {
                    double d = (candidatePos - center).norm();
                    if (d < distToClosestCluster) distToClosestCluster = d;
                }
            }

            if (distToClosestCluster > maxDistToExisting) {
                maxDistToExisting = distToClosestCluster;
                bestCandidateID = candidateID;
            }
        }

        if (bestCandidateID == -1) {
            failures++;
            if (failures > maxFailures) break; // 无法找到未覆盖点，退出
            continue; 
        }
        failures = 0; // 重置失败计数

        // 选定新中心
        const auto& newCenterPos = inMesh->vertices()[bestCandidateID].getCoords();
        const auto& newCenterVertex = inMesh->vertices()[bestCandidateID];
        // --- 步骤 B: 继承导引点特征 ---
        // 找到最近的导引点
        int nearestPilotIndex = pilotIndex.getClosestVertex(newCenterPos).index;
        const PilotPoint& nearestPilot = pilotPoints[nearestPilotIndex];

        // --- 步骤 C: 确定尺寸 ---
        // 查询最近的 targetVerticesPerCluster 个邻居
        auto kNearestVertexIDs = inputIndex.getClosestVertices(newCenterPos, targetVerticesPerCluster);
        
        double baseRadius = 0.0;
        if (!kNearestVertexIDs.empty()) {
            // 选择最远的一个邻居距离作为基础半径
            std::vector<double> squardRadius(kNearestVertexIDs.size());
            std::transform(kNearestVertexIDs.begin(), kNearestVertexIDs.end(), squardRadius.begin(), [&inMesh, bestCandidateID](auto i){
                return computeSquaredDifference(inMesh->vertex(i).rawCoords(), inMesh->vertex(bestCandidateID).rawCoords());
            });
            auto squardBaseRadius = std::max_element(squardRadius.begin(), squardRadius.end());
            baseRadius = std::sqrt(*squardBaseRadius);
        }
        
        // 防止半径过小
        PRECICE_ASSERT(baseRadius > 1e-6, "Anisotropic cluster radius is smaller than 1e-6");

        // --- 步骤 D: 创建各向异性簇 ---
        AnisotropicVertexCluster<RADIAL_BASIS_FUNCTION_T> newCluster(newCenterVertex, baseRadius, function, polynomial, inMesh, outMesh, nearestPilot, shapeParameter);
        
        clusters.push_back(newCluster);
        clusterCenters.push_back(newCenterPos);

        // --- 步骤 E: 更新覆盖状态 ---
        // 粗筛：利用 R-Tree 找新簇包围盒内的点
        // 精筛：用 Mahalanobis 距离判断
        double searchRadius;
        mesh::Mesh::VertexOffsets candidateIndices;
        if (nearestPilot.type == FeatureType::LINEAR) {
            // 面状为基础半径再压扁，体状为基础半径的圆球，因此最大搜索半径均为基础半径
            searchRadius = baseRadius * shapeParameter;
            candidateIndices = inputIndex.getClosestVertices(newCenterPos, searchRadius);

            for (int idx : candidateIndices) {
                if (isCovered[idx]) continue;

                if (newCluster.isCovering(inMesh->vertices()[idx].getCoords())) {
                    isCovered[idx] = true;
                    coveredCount++;
                }
            }
        } else if (nearestPilot.type == FeatureType::SURFACE) {
            // 面状为基础半径再压扁，体状为基础半径的圆球，因此最大搜索半径均为基础半径
            searchRadius = baseRadius;
            candidateIndices = inputIndex.getClosestVertices(newCenterPos, searchRadius);

            for (int idx : candidateIndices) {
                if (isCovered[idx]) continue;

                if (newCluster.isCovering(inMesh->vertices()[idx].getCoords())) {
                    isCovered[idx] = true;
                    coveredCount++;
                }
            }
        } else if (nearestPilot.type == FeatureType::VOLUMETRIC) {
            searchRadius = baseRadius;
            candidateIndices = inputIndex.getClosestVertices(newCenterPos, searchRadius);

            for (int idx : candidateIndices) {
                if (isCovered[idx]) continue;

                isCovered[idx] = true;
                coveredCount++;
            }
        }
        maxRange = std::max(maxRange, searchRadius);

    }

    return {clusters, maxRange};
}

} // namespace precice::mapping::impl