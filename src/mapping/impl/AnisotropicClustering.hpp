#pragma once

#include "mesh/Mesh.hpp"
#include "query/Index.hpp" // R-Tree
#include "logging/LogMacros.hpp"
#include "AnisotropicFeatures.hpp"
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

namespace precice::mapping::impl {

/**
 * @brief 生成各向异性子域 (研究点一·步骤二)
 * 
 * @param inputMesh 待覆盖的网格 (Source Mesh)
 * @param pilotPoints 第一步生成的导引点集
 * @param targetVerticesPerCluster 目标顶点数/cluster (用于计算基础半径)
 * @param mcSampleCount 蒙特卡洛采样的样本数 M (注意事项1)
 * @return std::vector<AnisotropicVertexCluster> 生成的簇列表
 */
std::vector<AnisotropicVertexCluster> generateAnisotropicSubdomains(
    const mesh::Mesh& inputMesh,
    const std::vector<PilotPoint>& pilotPoints,
    unsigned int targetVerticesPerCluster,
    unsigned int mcSampleCount = 10) 
{
    preciceTrace2("generateAnisotropicSubdomains", targetVerticesPerCluster, mcSampleCount);

    // 1. 准备数据结构
    std::vector<AnisotropicVertexCluster> clusters;
    int vertexCount = inputMesh.vertices().size();
    if (vertexCount == 0) return clusters;

    // 映射表/Bool数组标记覆盖状态 (注意事项1: 推荐使用vector<bool>以节省空间)
    std::vector<bool> isCovered(vertexCount, false);
    int coveredCount = 0;

    // 构建Pilot Points的查询索引，用于快速查找最近导引点 (注意事项2)
    // 这里需要将PilotPoint转换为preCICE可查询的格式，此处简化假设已有构建好的Index
    // 在实际集成中，可能需要临时构建一个只包含PilotPoint坐标的Mesh用于构建Tree
    mesh::Mesh pilotMesh("PilotMesh", 3, false);
    for(const auto& p : pilotPoints) {
        pilotMesh.createVertex(p.position);
    }
    query::Index pilotIndex(pilotMesh); // 构建R-Tree/KD-Tree
    pilotIndex.build();

    // 构建输入网格的查询索引，用于计算基础半径 (注意事项3)
    query::Index inputIndex(inputMesh); 
    inputIndex.build();

    // 随机数生成器
    std::mt19937 rng(12345); // 固定种子便于复现
    std::uniform_int_distribution<int> dist(0, vertexCount - 1);

    // 用于存储生成的Cluster中心，便于计算FPS距离
    // 注意：preCICE的query::Index不支持动态插入，所以我们可能需要线性扫描或分批重建
    // 为保证性能，FPS阶段若Cluster数量不多，可直接线性计算距离；若多，需优化。
    // 这里采用：维护一个简单的中心点列表
    std::vector<Eigen::Vector3d> clusterCenters;

    // 2. 迭代生成Cluster
    // 终止条件可设为覆盖率达99%或连续多次无法找到有效新簇
    int maxFailures = 100;
    int failures = 0;

    while (coveredCount < vertexCount) {
        // --- 步骤 A: 蒙特卡洛优化的 FPS 采样 (注意事项1) ---
        int bestCandidateID = -1;
        double maxDistToExisting = -1.0;

        // 抽取 M 个随机样本
        for (unsigned int i = 0; i < mcSampleCount; ++i) {
            int candidateID = dist(rng);
            
            // 如果已经被覆盖，跳过 (或寻找下一个未覆盖的)
            if (isCovered[candidateID]) continue;

            const auto& candidatePos = inputMesh.vertices()[candidateID].getCoords();

            // 计算该候选点到"现有簇"的最短距离
            double distToClosestCluster = std::numeric_limits<double>::max();
            
            if (clusterCenters.empty()) {
                // 第一个簇，距离无穷大，直接选中
                distToClosestCluster = std::numeric_limits<double>::max();
            } else {
                // 这里简化为线性扫描现有中心。
                // 随着簇数量增加，这里可以用 query::Index 优化，但考虑到FPS只需采样M个，
                // M * N_clusters 的开销通常可接受。
                for (const auto& center : clusterCenters) {
                    double d = (candidatePos - center).norm();
                    if (d < distToClosestCluster) distToClosestCluster = d;
                }
            }

            // FPS核心：选出"离现有簇最远"的那个候选点
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
        const auto& newCenterPos = inputMesh.vertices()[bestCandidateID].getCoords();

        // --- 步骤 B: 继承导引点特征 (注意事项2 & 7) ---
        // 找到最近的导引点
        int nearestPilotIndex = pilotIndex.getClosestVertexID(newCenterPos);
        const PilotPoint& nearestPilot = pilotPoints[nearestPilotIndex];

        // --- 步骤 C: 确定尺寸 (注意事项3) ---
        // 查询最近的 targetVerticesPerCluster 个邻居
        auto neighbors = inputIndex.getKNearestNeighborIDs(newCenterPos, targetVerticesPerCluster);
        
        double baseRadius = 0.0;
        if (!neighbors.empty()) {
            // 选择最远的一个邻居距离作为基础半径
            int furthestNeighborID = neighbors.back(); // preCICE返回的通常按距离排序，需确认API细节
            baseRadius = (inputMesh.vertices()[furthestNeighborID].getCoords() - newCenterPos).norm();
        }
        
        // 防止半径过小 (fallback)
        if (baseRadius < 1e-6) baseRadius = 0.01; // 需根据问题尺度调整

        // --- 步骤 D: 创建各向异性簇 (注意事项4 & 5) ---
        // 参数 3.0 是长轴放大倍数或短轴缩小倍数 (经验值)
        AnisotropicVertexCluster newCluster(newCenterPos, nearestPilot, baseRadius, 3.0);
        
        clusters.push_back(newCluster);
        clusterCenters.push_back(newCenterPos);

        // --- 步骤 E: 更新覆盖状态 (注意事项6) ---
        // 这里需要高效地找出新簇覆盖了哪些点。
        // 粗筛：利用 R-Tree 找新簇包围盒内的点
        // 精筛：用 Mahalanobis 距离判断
        
        // 估算最大搜索半径 (取最长轴)
        double searchRadius = baseRadius * 3.0; // 对应上面的 shapeParameter
        auto candidateIndices = inputIndex.getWindowIndices(newCenterPos, searchRadius);

        for (int idx : candidateIndices) {
            if (isCovered[idx]) continue;

            if (newCluster.isCovering(inputMesh.vertices()[idx].getCoords())) {
                isCovered[idx] = true;
                coveredCount++;
            }
        }
    }

    return clusters;
}

} // namespace precice::mapping::impl