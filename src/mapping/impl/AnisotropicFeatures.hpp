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

/**
 * @brief Voxel Downsampling(2D: split in 4 parts; 3D: split in 8 parts)
 * 
 * @param[in] inMesh input mesh(source mesh) for creating guide points
 * @param[in] bb bounding box of the input mesh
 * 
 * @return a collection of pilot points(mesh::Vertex) using for geometric feature computation
*/
Vertices samplePilotPoints(mesh::PtrMesh inMesh, const mesh::BoundingBox &bb)
{
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
                pilotPositions.emplace_back(mesh::Vertex({centerCoords, pilotPositions.size()}));
            }
        }
    } else if (dim == 3) {
        for (int i = 0; i < nSplits; ++i) {
            for (int j = 0; j < nSplits; ++j) {
                for (int k = 0; k < nSplits; ++k) {
                    centerCoords[0] = start[0] + i * distances[0];
                    centerCoords[1] = start[1] + j * distances[1];
                    centerCoords[2] = start[2] + k * distances[2];
                    pilotPositions.emplace_back(mesh::Vertex({centerCoords, pilotPositions.size()}));
                }
            }
        }
    }
    return pilotPositions;
}

/**
 * @brief Computes the geometric features around the pilot points using PCA.
 * 
 * @param[in] pilotPositions the collection of pilot points
 * @param[in] inMesh the mesh we search for the k-nearest vertices around each pilot point
 * 
 * @return a collection of LocalFeature structs representing the geometric features
 */
std::vector<PilotPoint> computeGeometricFeatures(const Vertices &pilotPositions, const mesh::PtrMesh inMesh)
{
    std::vector<PilotPoint> pilotPoints;
    const int kNearest = 20; // number of nearest neighbors to consider for PCA

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

/**
 * @brief Create ellipsoid clusters using local features around pilot points.
 *        All the clusters are splitted by pilot points, with anisotropic radii defined by the local features.
 *        Each pilot point has one feature shared by multiple clusters(oblate ellipsoid or prolate ellipsoid or sphere).
 * 
 * @param[in] inMesh The input mesh (input mesh for consistent, output mesh for conservative mappings), on which the
 *            clustering is computed. The input parameters \p verticesPerCluster and \p projectClustersToInput refer
 *            to the \p inMesh
 * @param[in] outMesh The output mesh (output mesh for consistent, input mesh for conservative mappings),
 * @param[in] relativeOverlap Value between zero and one, which steers the relative distance between cluster centers.
 *            A value of zero leads to no overlap, a value of one would lead to a complete overlap between clusters.
 * @param[in] verticesPerCluster Target number of vertices per partition. Used for estimating the cluster radius. Not
 *            used directly for creating the clusters.
 * @param[in] projectClustersToInput if enabled, moves the cluster centers to the closest vertex of the \p inMesh
 *
 * @return @todo describe return value
 */

} // namespace precice::mapping::impl