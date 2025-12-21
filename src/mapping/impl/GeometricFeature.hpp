#pragma once

#include <Eigen/Core>
#include <Eigen/src/Eigenvalues/SelfAdjointEigenSolver.h>
#include <numeric>

#include "mesh/Mesh.hpp"
#include "precice/impl/Types.hpp"

namespace precice::mapping::impl {
namespace {
using Vertices = std::vector<mesh::Vertex>;
}
enum FeatureType {
    LINEAR,
    SURFACE,
    VOLUMETRIC
};

/**
 * @brief Storage the geometric feature.
 * 
 * @param center the physical coordinates of the feature(the position of the guide point)
 * @param eigenvalues the eigenvalues(lamda1-3) after PCA decomposition, used to determine the feature type
 * @param eigenvectors the eigenvector matrix(principal directions), 
 *        used to determine the rotational pose of the anisotropic ellipsoid
 * @param type the type of the feature, with enumeration options:{LINEAR, SURFACE, VOLUMETRIC}
*/
struct LocalFeature{
    Eigen::Vector3d center;
    Eigen::Vector3d eigenvalues;
    Eigen::Matrix3d eigenvectors;
    FeatureType type;
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

    Vertices pilotPoints;
    std::vector<double> centerCoords(dim);
    // Fill the centers
    if (dim == 2) {
        for (int i = 0; i < nSplits; ++i) {
            for (int j = 0; j < nSplits; ++j) {
                centerCoords[0] = start[0] + i * distances[0];
                centerCoords[1] = start[1] + j * distances[1];
                pilotPoints.emplace_back(mesh::Vertex({centerCoords, pilotPoints.size()}));
            }
        }
    } else if (dim == 3) {
        for (int i = 0; i < nSplits; ++i) {
            for (int j = 0; j < nSplits; ++j) {
                for (int k = 0; k < nSplits; ++k) {
                    centerCoords[0] = start[0] + i * distances[0];
                    centerCoords[1] = start[1] + j * distances[1];
                    centerCoords[2] = start[2] + k * distances[2];
                    pilotPoints.emplace_back(mesh::Vertex({centerCoords, pilotPoints.size()}));
                }
            }
        }
    }
    return pilotPoints;
}

/**
 * @brief Computes the geometric features around the pilot points using PCA.
 * 
 * @param[in] pilotPoints the collection of pilot points
 * @param[in] inMesh the mesh we search for the k-nearest vertices around each pilot point
 * 
 * @return a collection of LocalFeature structs representing the geometric features
 */
std::vector<LocalFeature> computeGeometricFeatures(const Vertices &pilotPoints, const mesh::PtrMesh inMesh)
{
    std::vector<LocalFeature> features;
    const int kNearest = 20; // number of nearest neighbors to consider for PCA

    for (const auto &pilotPoint : pilotPoints) {
        // Step 1: find k-nearest neighbors
        auto neighborIDs = inMesh->index().getClosestVertices(pilotPoint.getCoords(), kNearest);
        PRECICE_ASSERT(!neighborIDs.empty(), "No neighbors found for pilot point ID {}.", pilotPoint.getID());

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
        PRECICE_ASSERT(eigSolver.info() == Eigen::Success, "PCA Eigen decomposition failed for pilot point ID {}.", pilotPoint.getID());
        Eigen::VectorXd eigenvalues = eigSolver.eigenvalues().reverse();
        Eigen::MatrixXd eigenvectors = eigSolver.eigenvectors().rowwise().reverse();

        // Step 4: determine feature type
        FeatureType featureType;
        if (inMesh->getDimensions() == 3) {
            if (eigenvalues(1) / eigenvalues(0) < 0.1 && eigenvalues(2) / eigenvalues(0) < 0.1) {
                featureType = LINEAR;
            } else if (eigenvalues(2) / eigenvalues(1) < 0.1) {
                featureType = SURFACE;
            } else {
                featureType = VOLUMETRIC;
            }
        } else { // 2D case
            if (eigenvalues(1) / eigenvalues(0) < 0.1) {
                featureType = LINEAR;
            } else {
                featureType = VOLUMETRIC;
            }
        }

        // Step 5: store the feature
        LocalFeature feature;
        feature.center = mean.head<3>();
        feature.eigenvalues = eigenvalues.head<3>();
        feature.eigenvectors = eigenvectors.leftCols(3);
        feature.type = featureType;
        features.push_back(feature);
    }
    return features;
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