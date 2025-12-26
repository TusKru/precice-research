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
GlobalAnisotropyParams computeGlobalAnisotropyParams(
    const mesh::PtrMesh inMesh,
    double baseRadius,
    const unsigned int kNeighbors = 20)
{
    GlobalAnisotropyParams params;
    Eigen::Matrix3d globalCov = Eigen::Matrix3d::Zero();
    int validPilots = 0;

    std::vector<Eigen::Vector3d> pilotPositions = samplePilotPositions(inMesh);
    
    // PCA over pilot positions
    for(const auto& p : pilotPositions) {
        auto neighbors = inMesh->index().getClosestVertices(p, kNeighbors);
        if(neighbors.size() < 4) continue; // Need at least 4 points for 3D covariance
        
        Eigen::MatrixXd data(neighbors.size(), 3);
        for(size_t i = 0; i < neighbors.size(); ++i) {
            const auto& v = inMesh->vertex(neighbors[i]);
            if(v.getDimensions() == 3) 
                data.row(i) = v.getCoords();
            else 
                data.row(i) << v.coord(0), v.coord(1), 0.0;
        }
        
        Eigen::Vector3d mean = data.colwise().mean();
        Eigen::MatrixXd centered = data.rowwise() - mean.transpose();
        Eigen::Matrix3d cov = (centered.adjoint() * centered) / double(data.rows() - 1);
        
        globalCov += cov;
        validPilots++;
    }
    
    if(validPilots > 0) 
        globalCov /= validPilots;
    else 
        globalCov = Eigen::Matrix3d::Identity(); // Fallback
    
    // Eigen Decomposition
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigSolver(globalCov);
    Eigen::Vector3d eigenvalues = eigSolver.eigenvalues(); // Sorted ascending
    Eigen::Matrix3d eigenvectors = eigSolver.eigenvectors();
    
    // Ensure eigenvalues are positive
    eigenvalues = eigenvalues.cwiseMax(1e-10);
    
    // Reverse to descending (lambda1 >= lambda2 >= lambda3)
    Eigen::Vector3d lambda;
    lambda << eigenvalues(2), eigenvalues(1), eigenvalues(0);
    Eigen::Matrix3d R;
    R << eigenvectors.col(2), eigenvectors.col(1), eigenvectors.col(0);
    
    params.rotation = R;
    
    // Determine semi-axes based on aspect ratio
    // Smallest axis = baseRadius.
    // axis_i = baseRadius * sqrt(lambda_i) / sqrt(lambda_min)
    // lambda_min is lambda(2)
    double min_sqrt_lambda = std::sqrt(lambda(2));
    
    params.semiAxes(0) = baseRadius * std::sqrt(lambda(0)) / min_sqrt_lambda;
    params.semiAxes(1) = baseRadius * std::sqrt(lambda(1)) / min_sqrt_lambda;
    params.semiAxes(2) = baseRadius;
    
    params.coverSearchRadius = params.semiAxes.maxCoeff();
    
    // Precompute Inverse Covariance: M = R * S^-2 * R^T
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
 * @return std::vector<Eigen::Vector3d> The points of the local lattice.
 */
std::vector<Eigen::Vector3d> generateLocalClusterCenters(
    const std::pair<Eigen::Vector3d, Eigen::Vector3d>& local_bounds,
    const Eigen::Vector3d& radii,
    double overlap)
{
    std::vector<Eigen::Vector3d> points;
    Eigen::Vector3d step = 2.0 * radii * 0.5;// (1.0 - overlap);
    
    if(step.minCoeff() <= 1e-9) return points;

    const auto& min = local_bounds.first;
    const auto& max = local_bounds.second;

    int nz = static_cast<int>(std::ceil((max.z() - min.z()) / step.z()));
    int ny = static_cast<int>(std::ceil((max.y() - min.y()) / step.y()));
    int nx = static_cast<int>(std::ceil((max.x() - min.x()) / step.x()));
    
    // Add buffer
    nz += 2; ny += 2; nx += 2;
    
    // Start a bit before min to cover edges after rotation/shifting
    double start_z = min.z() - step.z();
    double start_y = min.y() - step.y();
    double start_x = min.x() - step.x();

    for (int k = 0; k < nz; ++k) {
        double z = start_z + k * step.z();
        
        // Offset for layer k
        // FCC-like: Layer 0 at (0,0), Layer 1 at (0.5, 0.5)
        double layer_offset_y = (k % 2 != 0) ? step.y() * 0.5 : 0.0;
        double layer_offset_x = (k % 2 != 0) ? step.x() * 0.5 : 0.0;

        for (int j = 0; j < ny; ++j) {
            double y = start_y + j * step.y() + layer_offset_y;
            
            // Staggered rows: Row j+1 offset by 0.5 X
            double row_offset_x = (j % 2 != 0) ? step.x() * 0.5 : 0.0;
            
            double total_offset_x = layer_offset_x + row_offset_x;
            // normalize offset to be within [0, step.x)
            while(total_offset_x >= step.x()) total_offset_x -= step.x();

            for (int i = 0; i < nx; ++i) {
                double x = start_x + i * step.x() + total_offset_x;
                points.emplace_back(x, y, z);
            }
        }
    }
    return points;
}

} // namespace

/**
 * @brief Generate global anisotropic params and create Anisotropic Clustering with Global Params
 */
inline std::tuple<std::vector<Eigen::Vector3d>, GlobalAnisotropyParams> createAnisotropicClustering(
    const mesh::PtrMesh inMesh,
    const mesh::PtrMesh outMesh, 
    unsigned int targetVerticesPerCluster,
    double overlapRatio = 0.1) 
{
    precice::logging::Logger _log{"impl::createAnisotropicClustering"};
    // 1. Estimate Base Radius
    precice::mesh::BoundingBox localBB = inMesh->index().getRtreeBounds();
    double baseRadius = estimateClusterRadius(targetVerticesPerCluster, inMesh, localBB);
    
    // 2. Compute Global Anisotropy Params
    GlobalAnisotropyParams params = computeGlobalAnisotropyParams(inMesh, baseRadius);
    
    // 3. Generate Cluster Centers
    std::vector<Eigen::Vector3d> centers;
    int vertexCount = inMesh->vertices().size();
    if(vertexCount == 0) 
        return {centers, params};
    
    // Define Transformation
    Eigen::Vector3d center_mesh = localBB.center();
    Eigen::Matrix3d T_inv = params.rotation.transpose();
    
    // Calculate Local Generation Bounds
    Eigen::Vector3d local_min = Eigen::Vector3d::Constant(std::numeric_limits<double>::max());
    Eigen::Vector3d local_max = Eigen::Vector3d::Constant(std::numeric_limits<double>::lowest());
    
    const int dim = inMesh->getDimensions();
    
    for(int i=0; i<8; ++i) {
        Eigen::Vector3d p = Eigen::Vector3d::Zero();
        
        if (dim == 2 && i >= 4) break;

        p[0] = (i & 1) ? localBB.maxCorner()[0] : localBB.minCorner()[0];
        p[1] = (i & 2) ? localBB.maxCorner()[1] : localBB.minCorner()[1];
        if (dim == 3) {
            p[2] = (i & 4) ? localBB.maxCorner()[2] : localBB.minCorner()[2];
        }
        
        Eigen::Vector3d p_local = T_inv * (p - center_mesh);
        local_min = local_min.cwiseMin(p_local);
        local_max = local_max.cwiseMax(p_local);
    }
    
    // Add padding (1 radius)
    // Note: radii are axis aligned in local space!
    local_min -= params.semiAxes;
    local_max += params.semiAxes;
    
    // Generate positions of cluster centers(in local space)
    auto localPositions = generateLocalClusterCenters({local_min, local_max}, params.semiAxes, overlapRatio);
    
    // Transform to Global and Filter
    // Filter condition: discard any centers that are too far outside the original Mesh Bounds
    for(const auto& localPosition : localPositions) {
        // Transform cluster centers back to global space
        Eigen::Vector3d globalPosition = (params.rotation * localPosition) + center_mesh;
        
        if (inMesh->index().isAnyVertexInsideBox(mesh::Vertex({globalPosition, -1}), params.coverSearchRadius))
            centers.push_back(globalPosition);
    }
    
    return {centers, params};
}

} // namespace precice::mapping::impl
