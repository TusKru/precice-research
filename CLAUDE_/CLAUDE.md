# preCICE Partition of Unity Mapping - Anisotropic Algorithm Documentation

## Project Overview

This project implements a Partition of Unity Method (PUM) with Radial Basis Functions (RBF) for mesh-to-mesh data mapping in preCICE. The original implementation uses **spherical clusters**, while an **anisotropic (ellipsoidal) clustering** algorithm has been added as an extension.

## Architecture

### Class Hierarchy

```
Mapping (abstract base)
└── PartitionOfUnityMapping<RBF>
    ├── _clusters: vector<AnisotropicVertexCluster<RBF>>   (when _useAnisotropic=true)
    └── _clustersSpherical: vector<SphericalVertexCluster<RBF>> (original spherical)
```

### Key Files

| File | Purpose |
|------|---------|
| `src/mapping/Mapping.hpp` | Abstract base class for all mappings |
| `src/mapping/PartitionOfUnityMapping.hpp` | Main PUM mapping orchestrator |
| `src/mapping/RadialBasisFctSolver.hpp` | RBF system assembly and solving |
| `src/mapping/impl/CreateClustering.hpp` | Original spherical clustering |
| `src/mapping/impl/SphericalVertexCluster.hpp` | Original spherical cluster |
| `src/mapping/impl/AnisotropicClustering.hpp` | **Anisotropic clustering (user extension)** |
| `src/mapping/impl/AnisotropicVertexCluster.hpp` | **Anisotropic ellipsoidal cluster (user extension)** |
| `src/mapping/impl/BasisFunctions.hpp` | RBF basis functions (Wendland, etc.) |
| `src/mapping/impl/MappingDataCache.hpp` | JIT mapping data cache |

---

## PartitionOfUnityMapping Flow

### 1. `computeMapping()` - Cluster Setup

```
1. Determine inMesh/outMesh based on constraint (CONSERVATIVE vs CONSISTENT)
2. Create clustering:
   if (_useAnisotropic):
     createAnisotropicClustering() → GlobalAnisotropyParams + centerVertices
   else:
     createClustering() → clusterRadius + centerVertices
3. For each center:
   Create VertexCluster (Anisotropic or Spherical)
   - Query vertices inside cluster using R-tree index
   - Build RBF solver (assembles & decomposes matrix)
4. Compute normalized PU weights for each output vertex
5. exportClusterCentersAsVTU() (debug only)
```

### 2. `mapConsistent()` / `mapConservative()` - Data Mapping

```
For each cluster:
  1. Extract local input data (indexed by _inputIDs)
  2. Call _rbfSolver.solveConsistent/solveConservative
  3. Accumulate weighted results to output
```

### 3. Just-In-Time (JIT) Mapping

```
initializeMappingDataCache() → allocate polynomialContributions[], p[]
updateMappingDataCache() → compute RBF coefficients per cluster
mapConsistentAt() / mapConservativeAt() → evaluate at specific coordinates
completeJustInTimeMapping() → finalize conservative mapping
```

---

## Spherical Algorithm (Original)

### CreateClustering.hpp

**`estimateClusterRadius()`**: Estimates cluster radius from k-nearest-neighbor density
- Places random samples at bbox center and ±0.25*edge in each dimension
- Queries k nearest neighbors, takes max distance as local radius
- Returns median of sampled radii

**`createClustering()`**: Creates regular Cartesian grid of cluster centers
- Computes `clusterRadius` via `estimateClusterRadius()`
- Computes `maximumCenterDistance = sqrt(4/dim) * radius * (1 - overlap)`
- Creates grid of centers using z-curve indexing
- Tags empty clusters, projects to input mesh (optional), removes duplicates

### SphericalVertexCluster

**Geometry**: Sphere with center `c` and radius `r`

**Covering test**: Euclidean distance `(x-c)^T(x-c) <= r^2`

**Weight function**: `CompactPolynomialC2(radius)`
- Uses actual Euclidean distance
- Formula: `(1-p)^4 * (4p + 1)` where `p = distance/radius`

---

## Anisotropic Algorithm (User Extension)

### AnisotropicClustering.hpp

#### `GlobalAnisotropyParams` Structure

```cpp
struct GlobalAnisotropyParams {
  Eigen::Matrix3d rotation;        // R (eigenvectors from PCA)
  Eigen::Vector3d semiAxes;        // [a, b, c] semi-axis lengths
  Eigen::Matrix3d inverseCovariance; // M = R * diag(1/a²,1/b²,1/c²) * R^T
  double coverSearchRadius;        // max semi-axis (for coarse bbox filter)
};
```

#### `computeGlobalAnisotropyParams()`

**PCA-based anisotropy detection**:

1. **Pilot positions**: 4x4x4 (3D) or 4x4 (2D) voxel grid over mesh bbox
2. **For each pilot**:
   - Query neighbors within `2 * baseRadius`
   - Require ≥6 neighbors, else skip
   - Compute 3x3 covariance matrix of neighbor positions
   - Eigen-decomposition: eigenvalues λ₀≤λ₁≤λ₂, eigenvectors e₀,e₁,e₂
   - **2D**: Use λ₁ (min non-zero) and λ₂ (max); **3D**: Use λ₀ and λ₂
   - **Anisotropy ratio**: `sqrt(λ_max / λ_min)`
   - Filter: ratio must exceed **1.3** threshold
   - Accumulate: `globalCov += cov`, store eigenvectors

3. **Global eigen-decomposition** of `globalCov / validPilots`
4. **Coherence score**: Average `|localDir · globalDir|` across valid pilots
   - Range [0,1], higher = more aligned

5. **Dynamic ratio limiting**:
   - If `coherenceScore > 0.4`: allow higher ratios up to 2.5
   - Formula: `t = (score - 0.4) / 0.6`, `dynamicMaxRatio = 1.5 + t² * 1.0`

6. **Semi-axes computation**:
   - `geomRatio0 = sqrt(λ₂/λ₀)`, `geomRatio1 = sqrt(λ₁/λ₀)` (3D)
   - Apply `finalRatio = min(geomRatio, dynamicMaxRatio)`
   - Optionally override with `staticRatio1`, `staticRatio2`
   - Normalize: `norConstant = cbrt(r0*r1*1)` (volume-preserving)
   - `semiAxes = [baseRadius*r0/norConstant, baseRadius*r1/norConstant, baseRadius/norConstant]`

7. **Inverse covariance**: `M = R * diag(1/a²,1/b²,1/c²) * R^T`

#### `createClusterCenters()`

**Hexagonal Close Packing (FCC/HCP) lattice**:

1. Transform bbox to local ellipsoid coordinates: `p_local = R^T * (p - center)`
2. Compute local bounds with padding `semiAxes`
3. **Step size**: `step = coverage_factor * radii * (1 - overlap)`
   - 2D: `coverage_factor = sqrt(3)` (hexagonal)
   - 3D: `coverage_factor = sqrt(2)` (FCC)
4. **3D layer offset**: odd layers shift by `(0.5*step.x, 0.5*step.y)`
5. **2D row offset**: odd rows shift by `0.5*step.x`
6. Transform back: `globalPos = R * localPos + center`
7. Remove duplicates and empty clusters

#### `tagEmptyAnisotropicClusters()`

- Query vertices within `coverSearchRadius` (bounding sphere)
- Fine filter: check `(x-c)ᵀM(x-c) < 1` using `isCovering()`
- Tag if no vertices covered

#### `tagDuplicateProjectedCenters()`

- Grid-based spatial hash with cell size = `threshold`
- For each center, check 3x3x3 neighbor cells for duplicates
- Distance-based duplicate detection

### AnisotropicVertexCluster

**Geometry**: Ellipsoid with center `c`, semi-axes `[a,b,c]`, rotation `R`

**Covering test (Mahalanobis distance)**:
```cpp
bool isCovering(const Vertex& v) const {
    return computeD2(v) <= 1.0;  // (x-c)ᵀ * M * (x-c) <= 1
}

double computeD2(const Vertex& v) const {
    diff = v.pos - center;
    return diffᵀ * _inverseCovariance * diff;
}
```

**Weight function**: `CompactPolynomialC2(1.0)` with Mahalanobis distance
- Uses normalized distance `sqrt(d2)` where `d2 <= 1` inside ellipsoid
- Formula: `(1-p)^4 * (4p + 1)` where `p = sqrt(d2)`

---

## Key Differences: Spherical vs Anisotropic

| Aspect | Spherical | Anisotropic |
|--------|-----------|-------------|
| **Shape** | Sphere (radius `r`) | Ellipsoid (semi-axes `[a,b,c]`, rotation `R`) |
| **Covering test** | `(x-c)² <= r²` | `(x-c)ᵀM(x-c) <= 1` |
| **Weight distance** | Euclidean | Mahalanobis (normalized) |
| **Cluster creation** | Cartesian grid | FCC/HCP hexagonal lattice |
| **RBF matrix** | Euclidean `‖x-y‖` | Euclidean `‖x-y‖` |
| **Anisotropy params** | Single radius | Rotation + 3 semi-axes |
| **Params scope** | Per-cluster | Global (shared across all clusters) |

---

## Constructor Parameters

```cpp
PartitionOfUnityMapping(
    Mapping::Constraint,     // CONSISTENT or CONSERVATIVE
    int dimension,
    RBF_T function,          // e.g., CompactPolynomialC2
    Polynomial polynomial,   // ON, OFF, SEPARATE
    unsigned int verticesPerCluster,  // target vertices per cluster
    double relativeOverlap, // 0-1, controls cluster spacing
    bool projectToInput,    // snap centers to mesh vertices
    bool useDynamicRatio,   // use coherence-based ratio limiting
    bool useAnisotropic,    // enable anisotropic vs spherical
    double staticRatio1,    // override ratio0 if >= 1
    double staticRatio2     // override ratio1 if >= 1 (3D only)
);
```

---

## Optimization Considerations

### Current Performance Characteristics

1. **Cluster creation**: O(n_pilots * k_neighbors * d²) for PCA
2. **Vertex queries**: R-tree bbox query + fine ellipsoid filter
3. **RBF solver**: O(n³) matrix decomposition per cluster
4. **Mapping evaluation**: O(n_clusters * n_vertices_in_cluster²)

### Potential Bottlenecks in Anisotropic

1. **PCA at every pilot**: 64 pilots × eigen-decomposition = expensive
2. **Fine filtering**: `isCovering()` does matrix-vector multiplication per vertex
3. **Global params**: Single anisotropy for all clusters may not fit complex geometries
4. **Cluster count**: FCC packing may produce more clusters than Cartesian at boundaries

---

## Implemented Optimizations

### Static Ratio Default + Extreme Plate Detection (2024-03-29)

**Problem**:
1. Original algorithm with dynamic ratio limiting produces poor results for all mesh types
2. In parallel execution, local mesh patches from any geometry appear "plate-like" (ratio_21 ≈ 1)
3. Using geometry-aware detection based on ratio_21 fails because all local patches are detected as plate-like

**Solution**:
1. **Use static ratios st1=1.0, st2=3.0 as defaults** (matching the original best configuration)
2. **Detect extreme plate geometry** when GeomRatio > 10000, and fall back to spherical

**Code location**: `src/mapping/impl/AnisotropicClustering.hpp`, function `computeGlobalAnisotropyParams()`

**Behavior**:
```cpp
// Extreme plate detection: GeomRatio极大表示极端各向异性，应回退到球形
const double EXTREME_RATIO_THRESHOLD = 10000.0;
if (geomRatio0 > EXTREME_RATIO_THRESHOLD || geomRatio1 > EXTREME_RATIO_THRESHOLD) {
    params.fallbackToSpherical = true;
    return params;  // with isotropic parameters
}

// Use static ratios (1.0, 3.0 as defaults)
finalRatio0 = (staticRatio1 >= 1.0) ? staticRatio1 : 1.0;
finalRatio1 = (staticRatio2 >= 1.0) ? staticRatio2 : 3.0;
```

**Key insight**: In parallel execution, each process only sees a local patch. The global geometry type cannot be reliably determined from local PCA. Using static ratios is the most robust approach.

---

## Overall Conclusion

### Static Ratios (exp_large_fwd/rev)

With **static ratios (st1=1.0, st2=3.0)**, anisotropic with static ratios vs spherical:

| Mesh | Error Winner | Time Winner | Speedup | Verdict |
|------|--------------|-------------|---------|---------|
| **Tube** | ani (62-86%) | ani | 2-3.7x | **Anisotropic wins** |
| **Ellipsoid** | ani (60-86%) | ani | 1.3-1.6x | **Anisotropic wins** |
| **Plate** | sph (100%) | sph | 0.8-0.9x | **Spherical wins** |

### Pure Dynamic Ratios (exp_small_dynamic)

With **pure dynamic ratio limiting** (st=0.0, no static override):

| Mesh | Error | Time | Verdict |
|------|-------|------|---------|
| **Tube** | sph 8-46% better | sph 1.4-1.6x faster | **Spherical wins** |
| **Plate** | sph 64-121% better | sph 4-6x faster | **Spherical wins** |

### Default-Static Code (exp_small_default_static)

Code modified so static ratios 1.0/3.0 are the default:
- **Tube/Plate**: compares anisotropic (dynamic) vs spherical
- **Ellipsoid**: compares anisotropic (dynamic) vs anisotropic (static)

| Mesh | Error | Time | Verdict |
|------|-------|------|---------|
| **Tube** | sph 0-16% better | ani_dyn **1.4-1.8x faster** | Mixed (ani faster but less accurate) |
| **Ellipsoid** | mixed (ani_dyn 3/5) | ani_dyn **1.4-2.0x faster** | **Anisotropic wins** |
| **Plate** | sph 64-92% better | sph **3-5x faster** | **Spherical wins** |

**Key Findings**:
1. **Static ratios (st1=1.0, st2=3.0) are essential** - pure dynamic ratio limiting produces poor anisotropy
2. With proper static ratios, **Anisotropic excels on Tube/Ellipsoid** (faster + similar/better error)
3. **Plate remains problematic** for Anisotropic - sph wins on all metrics
4. **Dynamic vs Static on Ellipsoid** (exp_small_default_static): Dynamic ratio anisotropic is 1.4-2.0x faster than static ratio anisotropic

---

## Root Cause Analysis

### Eigenvalue Interpretation for Different Geometries

| Geometry | λ0 (min) | λ1 (mid) | λ2 (max) | ratio_21 | ratio_10 | Physical Meaning |
|----------|----------|----------|----------|----------|------------|------------------|
| **Tube** | short axis | short axis | long axis | >> 1 | ≈ 1 | One direction >> other two |
| **Ellipsoid** | short axis | mid axis | long axis | > 1 | > 1 | Three different axes |
| **Plate** | thickness | long axis | long axis | **≈ 1** | >> 1 | **Two directions >> third** |

### Problem: Geometry-Aware Ratio Assignment Not Sufficient for Plate

The current geometry-aware ratio assignment logic:

```cpp
if (ratio_21 < PLATE_THRESHOLD) {
    // Use max of (geomRatio0, geomRatio1) for both directions
    double ratioToUse = std::max(finalRatio0, finalRatio1);
    finalRatio0 = ratioToUse;
    finalRatio1 = ratioToUse;
} else {
    // Tube/Ellipsoid-like
    if (staticRatio1 >= 1.0) finalRatio0 = staticRatio1;
    if (staticRatio2 >= 1.0) finalRatio1 = staticRatio2;
}
```

**Issue**: Even with this logic, anisotropic still produces 47-68% higher error on Plate. The problem may be:

1. **FCC lattice mismatch**: FCC/HCP packing is optimized for isotropic spheres, not flat plates
2. **Global anisotropy assumption**: Single anisotropy params for all clusters may not capture local plate variations
3. **Weight function**: Mahalanobis distance normalization may not be optimal for thin geometries

---

## Conclusions & Hypotheses

### Confirmed Conclusions

1. **Anisotropic excels on elongated geometries (Tube, Ellipsoid) with static ratios**
   - Error: Anisotropic matches or slightly beats Spherical
   - Speed: 1.3-3.7x faster due to fewer clusters
   - Requires user-specified static ratios (st1=1.0, st2=3.0)

2. **Anisotropic underperforms on flat geometries (Plate) even with static ratios**
   - Error: 47-68% higher than Spherical with st1=1.0, st2=3.0
   - Speed: Slightly slower or similar
   - Problem persists despite geometry-aware ratio assignment

3. **Pure dynamic ratio limiting is insufficient (st=0.0)**
   - Produces poor anisotropy parameters
   - Anisotropic becomes **slower** than Spherical (not faster!)
   - Error is significantly worse (8-121% higher)
   - **Static ratios are essential for good performance**

4. **Dynamic ratio anisotropic is faster than static ratio anisotropic**
   - On Ellipsoid, dynamic is 1.4-2.0x faster than static (exp_small_default_static)
   - But still loses to Spherical on Plate
   - Trade-off: speed vs accuracy depending on geometry

### Hypotheses for Plate Underperformance

#### H1: FCC Lattice Suboptimal for Flat Geometries
- **Hypothesis**: FCC/HCP lattice is designed for close-packing of spheres, not for covering flat planes efficiently
- **Evidence**: Anisotropic produces more clusters than Spherical at fine resolutions on Plate
- **Test**: Try Cartesian grid (like Spherical) for Plate geometries

#### H2: Weight Function Distortion on Thin Geometries
- **Hypothesis**: Mahalanobis distance `sqrt((x-c)ᵀM(x-c))` over-normalizes in the thin direction, causing weight function to decay too sharply
- **Evidence**: Plate error gap increases with cluster size (avg_verts)
- **Test**: Compare with Euclidean distance weighted by local thickness

#### H3: Single Global Anisotropy Captures Only Major Axis
- **Hypothesis**: For Plate, the global rotation captures the in-plane orientation but not the thickness variation, leading to poor coverage at edges/corners
- **Evidence**: Error pattern suggests coverage issues rather than RBF approximation errors
- **Test**: Compare with locally computed anisotropy per cluster

#### H4: Cluster Center Projection Mismatch
- **Hypothesis**: Projecting cluster centers to input mesh vertices may cause misalignment between anisotropic ellipsoid orientation and actual vertex distribution
- **Evidence**: Plate error worse at finer resolutions where projection matters more
- **Test**: Disable `projectToInput` for anisotropic on Plate

#### H5: Dynamic Ratio Over-Normalizes Detected Anisotropy
- **Hypothesis**: Dynamic ratio limiting (without static overrides) produces overly aggressive anisotropy ratios, making ellipsoids too elongated
- **Evidence**: With st=0.0, anisotropic becomes slower (suggesting more, smaller clusters) and less accurate
- **Test**: Compare coherence scores between dynamic and static ratio experiments

### Recommended Next Experiments

| Experiment | Parameter | Expected Outcome |
|------------|-----------|------------------|
| Vary st1/st2 | Plate | Find better ratio for flat geometry |
| Dynamic + geometry-aware | Plate | Combine dynamic ratio with plate-specific logic |
| Cartesian vs FCC | Plate | Test if lattice type matters |
| Different overlap | Plate | 0.3, 0.4, 0.6 overlap values |
| Local anisotropy | All | Per-cluster PCA vs global |
| Disable projection | All | Test center snapping impact |

### Parallel Execution Note

In preCICE's parallel execution:
- Each process owns a local mesh partition (local + ghost vertices)
- `createAnisotropicClustering()` operates on **local mesh only**
- PCA is computed **independently per process** using local mesh bbox
- `getRtreeBounds()` returns **local** bounding box, not global

This local computation is acceptable since anisotropy parameters are used for local cluster geometry (covering test, weight computation), not for global mesh statistics.
