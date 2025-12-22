#pragma once

#include <Eigen/Core>
#include <vector>
#include <Eigen/src/Eigenvalues/SelfAdjointEigenSolver.h>
#include <numeric>

#include "mesh/Mesh.hpp"
#include "precice/impl/Types.hpp"

namespace precice::mapping::impl {

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


// 新增：各向异性簇类 (对应注意事项 7: 存储中心和导引点ID)
// 注意：虽然存储时尽量精简，但在运行时我们需要计算覆盖范围，
// 因此构造函数中会立即计算出形状矩阵。
class AnisotropicVertexCluster {
public:
    AnisotropicVertexCluster(const Eigen::Vector3d& center, 
                             const PilotPoint& pilot, 
                             double baseRadius,
                             double shapeParameter = 3.0) // shapeParameter用于拉伸/压缩比例
        : _center(center), _pilotID(pilot.id) 
    {
        PRECICE_ASSERT(baseRadius > 0, "r must be positive");
        PRECICE_ASSERT(shapeParameter > 0 && shapeParameter < 1, "shapeParameter must be (0, 1)");
        computeShapeMatrix(pilot, baseRadius, shapeParameter);
    }

    const Eigen::Vector3d& getCenter() const { return _center; }
    
    // 判断顶点是否在椭球覆盖范围内
    // 判据: (x-c)^T * M * (x-c) <= 1.0
    bool isCovering(const Eigen::Vector3d& vertex) const {
        Eigen::Vector3d diff = vertex - _center;
        return (diff.transpose() * _inverseCovariance * diff).value() <= 1.0;
    }

private:
    Eigen::Vector3d _center;
    int _pilotID;
    
    // 运行时缓存的逆协方差矩阵，用于快速判断覆盖 (Mahalanobis distance)
    Eigen::Matrix3d _inverseCovariance; 

    void computeShapeMatrix(const PilotPoint& pilot, double r, double alpha) {
        // r: 基础半径 (由k近邻距离决定)
        // alpha: 经验缩放系数 (注意事项4中的"固定经验值")
        
        Eigen::Vector3d semiAxes;
        Eigen::Matrix3d rotation = pilot.eigenvectors; // v1, v2, v3

        // 注意事项 4 & 5 的实现逻辑：
        switch (pilot.type) {
            case FeatureType::VOLUMETRIC: // 圆球
                semiAxes = Eigen::Vector3d(r, r, r);
                rotation = Eigen::Matrix3d::Identity(); // 无方向
                break;

            case FeatureType::LINEAR: // 长球 (Prolate)
                // 长轴(v1)放大，短轴(v2, v3)为基础半径
                semiAxes = Eigen::Vector3d(r / alpha, r, r);
                break;

            case FeatureType::SURFACE: // 扁球 (Oblate)
                // 法向v3压缩，切向v1/v2保持
                semiAxes = Eigen::Vector3d(r, r, r * alpha); 
                break;
        }

        // 构建形状矩阵 M = R * S^(-2) * R^T
        // 使得 d^2 = x^T M x <= 1
        Eigen::Matrix3d S_inv2 = Eigen::Matrix3d::Zero();
        S_inv2(0, 0) = 1.0 / (semiAxes[0] * semiAxes[0]);
        S_inv2(1, 1) = 1.0 / (semiAxes[1] * semiAxes[1]);
        S_inv2(2, 2) = 1.0 / (semiAxes[2] * semiAxes[2]);

        _inverseCovariance = rotation * S_inv2 * rotation.transpose();
    }
};

} // namespace precice::mapping::impl