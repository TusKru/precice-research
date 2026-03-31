# Anisotropic Partition of Unity Mapping - Algorithm Notes for Paper Writing

## 1. Problem Statement

Mesh-to-mesh data mapping is fundamental in multiphysics coupling. Traditional Radial Basis Function (RBF) based Partition of Unity Method (PUM) uses **spherical clusters**, which are inefficient for elongated or anisotropic geometries (e.g., tubes, ellipsoids).

**Goal**: Extend the spherical PUM-RBF approach to anisotropic (ellipsoidal) clusters that better adapt to geometry shape, reducing cluster count and improving efficiency.

---

## 2. Core Algorithm Overview

### 2.1 Partition of Unity Method

The mapping reconstructs a field $\tilde{u}(\mathbf{x})$ as a weighted sum of local approximations:

$$\tilde{u}(\mathbf{x}) = \sum_{k=1}^{K} w_k(\mathbf{x}) \cdot u_k(\mathbf{x})$$

where:
- $K$ = number of clusters covering point $\mathbf{x}$
- $w_k(\mathbf{x})$ = partition of unity weight (Shepard's method)
- $u_k(\mathbf{x})$ = local RBF approximation in cluster $k$

**Constraint**: $\sum_{k=1}^{K} w_k(\mathbf{x}) = 1$

### 2.2 Spherical vs Anisotropic Clusters

| Aspect | Spherical | Anisotropic |
|--------|-----------|-------------|
| Shape | Sphere (radius $r$) | Ellipsoid (semi-axes $[a, b, c]$, rotation $R$) |
| Covering test | $\lVert \mathbf{x} - \mathbf{c} \rVert^2 \leq r^2$ | $(\mathbf{x}-\mathbf{c})^T M (\mathbf{x}-\mathbf{c}) \leq 1$ |
| Weight distance | Euclidean | Mahalanobis (normalized) |
| Cluster lattice | Cartesian grid | FCC/HCP hexagonal lattice |
| Parameter scope | Per-cluster radius | Global (shared rotation + semi-axes) |

---

## 3. Algorithm Pipeline

### Phase 1: Global Anisotropy Parameter Computation

**Input**: Input mesh vertices
**Output**: Global anisotropy parameters $(R, [a, b, c], M)$

#### Step 1: Pilot Position Sampling
- Create a **voxel grid** over the mesh bounding box
- Resolution: 4×4×4 (3D) or 4×4 (2D) = 64 or 16 pilot points
- Pilot positions at voxel centers

#### Step 2: Local PCA at Each Pilot
For each pilot position $\mathbf{p}$:
1. Query $k$ nearest neighbors within search radius $r_{search} = 2 \cdot r_{base}$
2. Require $\geq 6$ neighbors (numerical stability)
3. Compute 3×3 covariance matrix $C$ of neighbor positions
4. Eigen-decomposition: $C = V \Lambda V^T$, eigenvalues $\lambda_0 \leq \lambda_1 \leq \lambda_2$

**Anisotropy ratio per pilot**:
$$r_{local} = \sqrt{\frac{\lambda_{max}}{\lambda_{min}}}$$

Filter pilots with $r_{local} < 1.3$ (isotropic regions).

#### Step 3: Global Eigen-Decomposition
Aggregate valid pilot covariances:
$$C_{global} = \frac{1}{N_{valid}} \sum_{i=1}^{N_{valid}} C_i$$

Eigen-decompose $C_{global}$ to obtain:
- **Rotation matrix** $R$ (eigenvectors, sorted $\lambda_2, \lambda_1, \lambda_0$)
- **Global anisotropy ratios**: $r_0 = \sqrt{\lambda_2/\lambda_0}$, $r_1 = \sqrt{\lambda_1/\lambda_0}$

#### Step 4: Coherence Score (Dynamic Ratio Limiting)
$$s_{coherence} = \frac{1}{N_{valid}} \sum_{i=1}^{N_{valid}} |\mathbf{d}_{global} \cdot \mathbf{d}_{local,i}|$$

where $\mathbf{d}_{global}$ is the major eigenvector. Range [0,1], higher = more aligned.

**Dynamic ratio limit**:
$$r_{max}(s) = \begin{cases} 1.5 + t^2 & \text{if } s > 0.4 \\ 1.0 & \text{otherwise} \end{cases}$$
where $t = (s - 0.4) / 0.6$

#### Step 5: Semi-Axis Computation

**Volume-preserving normalization**:
$$n = \sqrt[3]{r_0 \cdot r_1 \cdot 1}$$

**Semi-axes**:
$$a = r_{base} \cdot \frac{r_0}{n}, \quad b = r_{base} \cdot \frac{r_1}{n}, \quad c = r_{base} \cdot \frac{1}{n}$$

#### Step 6: Inverse Covariance Matrix
$$M = R \cdot \text{diag}(1/a^2, 1/b^2, 1/c^2) \cdot R^T$$

This enables efficient Mahalanobis distance: $d^2(\mathbf{x}) = (\mathbf{x}-\mathbf{c})^T M (\mathbf{x}-\mathbf{c})$

#### Critical Insight: Static Ratios
**Finding**: Pure dynamic ratio limiting produces poor results in parallel execution because local mesh patches appear isotropic. **Static ratios $r_0=1.0, r_1=3.0$ are essential defaults**.

**Extreme plate detection**: If $r_0 > 10000$ or $r_1 > 10000$, fallback to spherical (numerical issues with extreme anisotropy).

---

### Phase 2: Cluster Center Generation

#### Hexagonal Close Packing (FCC/HCP) Lattice

**Motivation**: Better space-filling than Cartesian grid for ellipsoidal clusters.

**Algorithm**:
1. Transform bbox to local ellipsoid coordinates: $\mathbf{p}_{local} = R^T (\mathbf{p} - \mathbf{c}_{bbox})$
2. Compute local bounds with padding $[a, b, c]$
3. Step size: $s = c_{factor} \cdot [a, b, c] \cdot (1 - \alpha)$
   - $c_{factor} = \sqrt{2}$ (3D, FCC), $\sqrt{3}$ (2D, hexagonal)
   - $\alpha$ = overlap ratio
4. **3D FCC layer offset**: odd layers shift by $(0.5 s_x, 0.5 s_y)$
5. **Row offset**: odd rows shift by $0.5 s_x$
6. Transform back to global: $\mathbf{p}_{global} = R \cdot \mathbf{p}_{local} + \mathbf{c}_{bbox}$

**Coverage factor derivation**:
- FCC: $\sqrt{2} \approx 1.414$ (close-packing of spheres)
- Hexagonal: $\sqrt{3} \approx 1.732$ (close-packing of circles)

---

### Phase 3: Cluster Filtering

#### Empty Cluster Tagging
1. **Coarse filter**: R-tree bbox query using $a$ (max semi-axis) as radius
2. **Fine filter**: Mahalanobis test $(x-c)^T M (x-c) < 1$
3. Tag cluster as empty if no vertices pass

#### Duplicate Removal (Post-Projection)
1. Project centers to nearest mesh vertices (if enabled)
2. Grid-based spatial hash with cell size $= 0.4 \cdot \min(s)$
3. Check 3×3×3 neighbor cells for duplicates
4. Tag duplicates for removal

---

### Phase 4: Local RBF Solver Construction

For each non-empty cluster:
1. Query input/output vertices inside ellipsoid (R-tree + fine filter)
2. Build RBF system with Polynomial = SEPARATE
3. Compute matrix decomposition (for solve efficiency)

**RBF System** (consistent mapping):
$$\begin{bmatrix} \Phi & P \\ P^T & 0 \end{bmatrix} \begin{bmatrix} c \\ \lambda \end{bmatrix} = \begin{bmatrix} u \\ 0 \end{bmatrix}$$

where $\Phi_{ij} = \phi(\lVert \mathbf{x}_i - \mathbf{x}_j \rVert)$, $P$ = polynomial basis.

---

### Phase 5: Partition of Unity Weight Computation

For each output vertex $\mathbf{v}$:

1. **Find covering clusters** (R-tree bbox + ellipsoid test)
2. **Compute raw weights** using CompactPolynomialC2:
$$w_k = \phi_{CP}(\sqrt{d_k^2}), \quad d_k^2 = (\mathbf{v} - \mathbf{c}_k)^T M_k (\mathbf{v} - \mathbf{c}_k)$$

where $\phi_{CP}(r) = (1-r)^4 (4r + 1)$ for $r \leq 1$, else $0$.

3. **Normalize**: $w_k^{norm} = w_k / \sum_j w_j$

---

### Phase 6: Data Mapping

**Consistent mapping**:
$$\tilde{u}(\mathbf{x}) = \sum_{k} w_k^{norm} \cdot u_k(\mathbf{x})$$

**Conservative mapping**:
$$\int_{\Omega} \tilde{u}(\mathbf{x}) d\Omega = \int_{\Omega} u(\mathbf{x}) d\Omega$$

Solved via local RBF conservative constraint + weight accumulation.

---

## 4. Key Mathematical Formulas

### Mahalanobis Distance
$$d_M^2(\mathbf{x}, \mathbf{c}, M) = (\mathbf{x} - \mathbf{c})^T M (\mathbf{x} - \mathbf{c})$$

Ellipsoid equation: $d_M^2(\mathbf{x}, \mathbf{c}, M) = 1$

### CompactPolynomialC2 Basis Function
$$\phi_{CP}(r) = \begin{cases} (1-r)^4 (4r + 1) & \text{if } r \leq 1 \\ 0 & \text{otherwise} \end{cases}$$

### Weight Function (Anisotropic)
$$w(\mathbf{x}) = \phi_{CP}\left(\sqrt{(\mathbf{x}-\mathbf{c})^T M (\mathbf{x}-\mathbf{c})}\right)$$

### PCA Covariance
$$C = \frac{1}{N-1} \sum_{i=1}^N (\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^T$$

---

## 5. Algorithm Complexity

| Phase | Complexity | Notes |
|-------|------------|-------|
| Pilot PCA | $O(N_{pilots} \cdot k \cdot d^2)$ | $d=3$, $k$ neighbors |
| Cluster center generation | $O(N_{centers})$ | FCC lattice |
| Vertex queries | $O(N_{clusters} \cdot \log N)$ | R-tree |
| RBF matrix decomposition | $O(N_{clusters} \cdot n_v^3)$ | Per cluster, $n_v$ = vertices/cluster |
| Mapping evaluation | $O(N_{clusters} \cdot n_v^2)$ | Matrix-vector multiply |

---

## 6. Performance Characteristics

### When Anisotropic Outperforms Spherical
- **Elongated geometries**: Tube, Ellipsoid
- **Static ratios**: $r_0=1.0, r_1=3.0$ (critical)
- **Speedup**: 1.3-3.7× faster due to fewer clusters

### When Anisotropic Underperforms
- **Flat geometries**: Plate
- **Issue**: FCC lattice suboptimal for thin geometries
- **Error**: 47-68% higher than spherical

### Critical Parameters
| Parameter | Default | Effect |
|-----------|---------|--------|
| `staticRatio1` | 1.0 | Controls elongation in minor axis direction |
| `staticRatio2` | 3.0 | Controls elongation in major axis direction |
| `verticesPerCluster` | 50 | Target cluster size |
| `relativeOverlap` | 0.15 | Cluster spacing |

---

## 7. Novel Contributions (For Paper)

1. **Global PCA-based anisotropy detection** with pilot voxel sampling
2. **Volume-preserving ellipsoid construction** with static ratio defaults
3. **FCC/HCP lattice for ellipsoidal clusters** (vs Cartesian for spherical)
4. **Mahalanobis distance covering test** for proper ellipsoid geometry
5. **Extreme anisotropy fallback** to spherical for numerical stability
6. **Coherence-based dynamic ratio limiting** for aligned geometries

---

## 8. Open Questions / Future Work

1. **Per-cluster anisotropy**: Local PCA instead of global (addresses plate geometry)
2. **Alternative lattices**: Cartesian grid for flat geometries
3. **Adaptive overlap**: Vary overlap based on local curvature
4. **Weight function**: Compare Mahalanobis vs Euclidean for thin geometries

---

## 9. References to Code

| Component | File | Key Functions |
|-----------|------|---------------|
| Anisotropy detection | `AnisotropicClustering.hpp:85-313` | `computeGlobalAnisotropyParams()` |
| Pilot sampling | `AnisotropicClustering.hpp:29-71` | `samplePilotPositions()` |
| Center generation | `AnisotropicClustering.hpp:324-446` | `createClusterCenters()` |
| Empty cluster tagging | `AnisotropicClustering.hpp:510-538` | `tagEmptyAnisotropicClusters()` |
| Duplicate removal | `AnisotropicClustering.hpp:561-630` | `tagDuplicateProjectedCenters()` |
| Ellipsoid cluster | `AnisotropicVertexCluster.hpp:26-125` | Class definition |
| Covering test | `AnisotropicVertexCluster.hpp:269-314` | `isCovering()`, `computeD2()` |
| Weight computation | `AnisotropicVertexCluster.hpp:316-331` | `computeWeight()` |
| Main orchestrator | `PartitionOfUnityMapping.hpp:183-302` | `computeMapping()` |
| PU weight computation | `PartitionOfUnityMapping.hpp:305-363` | `computeNormalizedWeight()` |

---

## 10. Experimental Results Summary

From CLAUDE.md experiments:

| Geometry | Anisotropic Speedup | Error vs Spherical | Recommendation |
|-----------|---------------------|-------------------|----------------|
| Tube | 2-3.7× faster | 62-86% better | **Use Anisotropic** |
| Ellipsoid | 1.3-1.6× faster | 60-86% better | **Use Anisotropic** |
| Plate | 0.8-0.9× slower | 47-68% worse | Use Spherical |

**Key finding**: Static ratios (1.0, 3.0) are essential. Pure dynamic ratio limiting fails in parallel execution.
