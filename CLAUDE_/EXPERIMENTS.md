# Anisotropic Clustering - Experimental Results

## Experiment Metadata

| Experiment ID | Description | Mesh | n_src | n_tgt | Size | Directory |
|--------------|-------------|------|-------|-------|------|-----------|
| exp_large_fwd | Static ratios forward | Tube/Ellipsoid/Plate | 4.5M | 5.0M | 280x280 | `exp_large_fwd/` |
| exp_large_rev | Static ratios reverse | Tube/Ellipsoid/Plate | 5.0M | 4.5M | 280x280 | `exp_large_rev/` |
| exp_small_dynamic | Pure dynamic ratios | Tube/Plate | 0.9M | 1.0M | 280x280 | `exp_small_dynamic/` |
| exp_small_default_static | Code modified: default=static | Tube/Ellipsoid/Plate | 0.9M | 1.0M | 280x280 | `exp_small_default_static/` |

**Parameter Notes**:
- **static ratios**: Filename contains `st1.0to3.0` → use static ratio1=1.0, ratio2=3.0
- **dynamic**: Filename contains `dynamic` → algorithm dynamically determines ratios (no static override)
- **default_static**: Filename contains `default_static` → code was modified so static ratios 1.0/3.0 are the default

**Common Configuration**:
- RBF basis function: CompactPolynomialC2
- Polynomial: SEPARATE
- Overlap: 0.50
- Benchmark method: ASTE error benchmarking tool

**version labels**:
- `dev` = anisotropic variant
- `ori` = baseline (spherical, or static-ratio anisotropic for Ellipsoid comparison)

**Filename Format**:
```
error_benchmark_<mesh>_<n_src>_<n_tgt>_<size>[_st<st1>to<st2>|_dynamic|default_static].csv
```

---

## Error Comparison (avg_verts controlled)

Comparison criterion: at **similar avg_verts** (within tolerance=5), lower error wins.

### exp_large_fwd (Static Ratios 1.0/3.0, ani vs sph)

| Mesh | ani_avg | ani_error | sph_avg | sph_error | diff% | Winner |
|------|---------|-----------|---------|-----------|-------|--------|
| **Tube** | 50 | 8.91e-06 | 52 | 9.46e-06 | -5.8% | ani |
| **Tube** | 65 | 1.10e-05 | 63 | 1.11e-05 | -0.5% | ani |
| **Tube** | 76 | 1.31e-05 | 78 | 1.37e-05 | -3.8% | ani |
| **Tube** | 91 | 1.50e-05 | 93 | 1.63e-05 | -7.7% | ani |
| **Tube** | 111 | 1.95e-05 | 107 | 1.85e-05 | +5.7% | sph |
| **Tube** | 121 | 2.13e-05 | 120 | 2.07e-05 | +2.9% | sph |
| **Tube** | 135 | 2.29e-05 | 138 | 2.39e-05 | -4.2% | ani |
| **Tube** | 150 | 2.61e-05 | 150 | 2.60e-05 | +0.4% | sph |
| **Ellipsoid** | 55 | 3.77e-03 | 50 | 3.15e-03 | +19.7% | sph |
| **Ellipsoid** | 77 | 5.17e-03 | 76 | 5.30e-03 | -2.5% | ani |
| **Ellipsoid** | 89 | 5.98e-03 | 89 | 6.44e-03 | -7.2% | ani |
| **Ellipsoid** | 101 | 6.86e-03 | 103 | 7.52e-03 | -8.9% | ani |
| **Ellipsoid** | 123 | 8.48e-03 | 118 | 8.69e-03 | -2.4% | ani |
| **Ellipsoid** | 136 | 9.41e-03 | 132 | 9.81e-03 | -4.0% | ani |
| **Ellipsoid** | 148 | 1.01e-02 | 144 | 1.08e-02 | -5.7% | ani |
| **Plate** | 75 | 6.29e-05 | 78 | 4.24e-05 | **+48.4%** | sph |
| **Plate** | 88 | 7.54e-05 | 90 | 5.06e-05 | **+48.8%** | sph |
| **Plate** | 109 | 9.33e-05 | 113 | 6.12e-05 | **+52.4%** | sph |
| **Plate** | 117 | 9.97e-05 | 113 | 6.12e-05 | **+62.9%** | sph |

### exp_large_rev (Static Ratios 1.0/3.0, ani vs sph)

| Mesh | ani_avg | ani_error | sph_avg | sph_error | diff% | Winner |
|------|---------|-----------|---------|-----------|-------|--------|
| **Tube** | 52 | 7.34e-06 | 51 | 7.67e-06 | -4.4% | ani |
| **Tube** | 64 | 8.93e-06 | 59 | 9.01e-06 | -0.9% | ani |
| **Tube** | 78 | 1.05e-05 | 73 | 1.11e-05 | -5.3% | ani |
| **Tube** | 89 | 1.21e-05 | 88 | 1.32e-05 | -7.7% | ani |
| **Tube** | 114 | 1.43e-05 | 113 | 1.68e-05 | -14.6% | ani |
| **Tube** | 115 | 1.64e-05 | 113 | 1.68e-05 | -2.4% | ani |
| **Tube** | 130 | 1.96e-05 | 131 | 1.93e-05 | +1.5% | sph |
| **Tube** | 155 | 2.09e-05 | 156 | 2.30e-05 | -8.9% | ani |
| **Ellipsoid** | 65 | 4.06e-03 | 60 | 3.70e-03 | +9.8% | sph |
| **Ellipsoid** | 76 | 4.67e-03 | 75 | 4.61e-03 | +1.5% | sph |
| **Ellipsoid** | 88 | 5.42e-03 | 88 | 5.62e-03 | -3.5% | ani |
| **Ellipsoid** | 100 | 6.28e-03 | 103 | 6.72e-03 | -6.6% | ani |
| **Ellipsoid** | 133 | 8.44e-03 | 132 | 8.66e-03 | -2.5% | ani |
| **Ellipsoid** | 146 | 9.40e-03 | 143 | 9.74e-03 | -3.5% | ani |
| **Plate** | 74 | 5.56e-05 | 78 | 3.77e-05 | **+47.6%** | sph |
| **Plate** | 90 | 6.87e-05 | 90 | 4.42e-05 | **+55.3%** | sph |
| **Plate** | 111 | 8.57e-05 | 112 | 5.45e-05 | **+57.3%** | sph |
| **Plate** | 130 | 1.05e-04 | 128 | 6.21e-05 | **+68.4%** | sph |

### exp_small_dynamic (Pure Dynamic Ratios, ani vs sph)

Both `dev` (ani) and `ori` (sph) use dynamic ratio mode (st=0.0, no static override):

| Mesh | ani_avg | ani_error | sph_avg | sph_error | diff% | Winner |
|------|---------|-----------|---------|-----------|-------|--------|
| **Tube** | 109 | 1.70e-04 | 106 | 1.57e-04 | +8.7% | sph |
| **Tube** | 123 | 2.54e-04 | 117 | 1.74e-04 | **+46.0%** | sph |
| **Tube** | 144 | 3.14e-04 | 148 | 2.19e-04 | **+43.4%** | sph |
| **Tube** | 177 | 3.68e-04 | 174 | 2.54e-04 | **+45.1%** | sph |
| **Plate** | 98 | 5.58e-04 | 91 | 2.52e-04 | **+121.1%** | sph |
| **Plate** | 136 | 6.23e-04 | 128 | 3.59e-04 | **+73.3%** | sph |
| **Plate** | 142 | 7.65e-04 | 128 | 3.59e-04 | **+112.9%** | sph |
| **Plate** | 172 | 9.17e-04 | 165 | 4.58e-04 | **+100.4%** | sph |

### exp_small_default_static (Code: Default=Static, ani vs sph/sani)

Code was modified so static ratios 1.0/3.0 are the default:
- **Tube/Plate**: `dev` (ani) vs `ori` (sph), both use dynamic ratios
- **Ellipsoid**: `dev` (ani, dynamic) vs `ori` (ani, static 1.0to3.0)

| Mesh | dev_avg | dev_error | ori_avg | ori_error | diff% | Winner | Comparison |
|------|---------|-----------|---------|-----------|-------|--------|------------|
| **Tube** | 48 | 7.70e-05 | 51 | 7.93e-05 | -2.9% | ani | ani_dyn vs sph |
| **Tube** | 73 | 1.16e-04 | 75 | 1.15e-04 | +0.7% | sph | ani_dyn vs sph |
| **Tube** | 92 | 1.45e-04 | 91 | 1.37e-04 | +6.0% | sph | ani_dyn vs sph |
| **Tube** | 112 | 2.02e-04 | 117 | 1.74e-04 | +15.9% | sph | ani_dyn vs sph |
| **Tube** | 133 | 2.16e-04 | 136 | 2.00e-04 | +8.0% | sph | ani_dyn vs sph |
| **Ellipsoid** | 62 | 1.89e-02 | 59 | 1.88e-02 | +0.8% | sani | ani_dyn vs ani_static |
| **Ellipsoid** | 87 | 2.79e-02 | 80 | 2.43e-02 | +14.9% | sani | ani_dyn vs ani_static |
| **Ellipsoid** | 110 | 3.57e-02 | 120 | 4.13e-02 | -13.4% | ani_dyn | ani_dyn vs ani_static |
| **Ellipsoid** | 136 | 4.50e-02 | 138 | 4.62e-02 | -2.7% | ani_dyn | ani_dyn vs ani_static |
| **Ellipsoid** | 171 | 5.46e-02 | 178 | 6.31e-02 | -13.5% | ani_dyn | ani_dyn vs ani_static |
| **Plate** | 97 | 4.84e-04 | 91 | 2.52e-04 | **+91.6%** | sph | ani_dyn vs sph |
| **Plate** | 137 | 6.47e-04 | 128 | 3.59e-04 | **+80.1%** | sph | ani_dyn vs sph |
| **Plate** | 168 | 8.10e-04 | 165 | 4.58e-04 | **+77.0%** | sph | ani_dyn vs sph |
| **Plate** | 208 | 9.50e-04 | 204 | 5.47e-04 | **+73.8%** | sph | ani_dyn vs sph |
| **Plate** | 255 | 1.10e-03 | 254 | 6.67e-04 | **+64.4%** | sph | ani_dyn vs sph |

Legend: `sph` = spherical, `sani` = anisotropic with static ratios, `ani_dyn` = anisotropic with dynamic ratios

### Error Summary

| Mesh | exp_large_fwd | exp_large_rev | exp_small_dynamic | exp_small_default_static |
|------|---------------|---------------|------------------|--------------------------|
| **Tube** | ani 5/8 | ani 6/7 | sph 0/4 | sph 4/5 |
| **Ellipsoid** | ani 6/7 | ani 3/5 | N/A | ani_dyn 3/5 (vs sani) |
| **Plate** | sph 0/4 | sph 0/4 | sph 0/4 | sph 0/5 |

---

## Time Comparison (computeMapping, avg_verts controlled)

### exp_large_fwd Time

| Mesh | ani_avg | ani_time(μs) | sph_avg | sph_time(μs) | Speedup |
|------|---------|--------------|---------|--------------|---------|
| **Tube** | 50 | 751,629 | 52 | 1,526,653 | **2.0x** |
| **Tube** | 76 | 861,427 | 78 | 3,147,177 | **3.7x** |
| **Tube** | 150 | 1,614,688 | 150 | 3,815,309 | **2.4x** |
| **Ellipsoid** | 89 | 42,527,592 | 89 | 63,063,571 | **1.5x** |
| **Ellipsoid** | 148 | 102,758,970 | 144 | 156,763,836 | **1.5x** |
| **Plate** | 75 | 691,326 | 78 | 831,059 | **1.2x** |
| **Plate** | 88 | 968,012 | 90 | 28,467,954 | **29x** (outlier) |
| **Plate** | 109 | 991,548 | 113 | 883,779 | **0.89x** |
| **Plate** | 117 | 1,066,334 | 113 | 883,779 | **0.83x** |

### exp_large_rev Time

| Mesh | ani_avg | ani_time(μs) | sph_avg | sph_time(μs) | Speedup |
|------|---------|--------------|---------|--------------|---------|
| **Tube** | 52 | 644,373 | 51 | 1,612,469 | **2.5x** |
| **Tube** | 155 | 1,342,230 | 156 | 3,509,891 | **2.6x** |
| **Ellipsoid** | 88 | 52,180,866 | 88 | 68,754,062 | **1.3x** |
| **Ellipsoid** | 146 | 110,959,199 | 143 | 178,113,819 | **1.6x** |
| **Plate** | 74 | 763,852 | 78 | 657,210 | **0.86x** |
| **Plate** | 90 | 924,736 | 90 | 738,076 | **0.80x** |
| **Plate** | 130 | 1,257,822 | 128 | 1,002,420 | **0.80x** |

### exp_small_dynamic Time

Both use dynamic ratios:

| Mesh | ani_avg | ani_time(μs) | sph_avg | sph_time(μs) | Speedup |
|------|---------|--------------|---------|--------------|---------|
| **Tube** | 109 | 687,642 | 106 | 535,540 | **0.78x** |
| **Tube** | 123 | 929,402 | 117 | 575,235 | **0.62x** |
| **Tube** | 144 | 1,059,549 | 148 | 746,187 | **0.70x** |
| **Tube** | 177 | 1,361,085 | 174 | 938,905 | **0.69x** |
| **Plate** | 98 | 953,865 | 91 | 209,743 | **0.22x** |
| **Plate** | 136 | 1,331,918 | 128 | 259,919 | **0.20x** |
| **Plate** | 172 | 1,852,785 | 165 | 328,402 | **0.18x** |
| **Plate** | 219 | 3,200,181 | 229 | 506,237 | **0.16x** |

### exp_small_default_static Time

| Mesh | dev_avg | dev_time | ori_avg | ori_time | Speedup | Comparison |
|------|---------|----------|---------|----------|---------|------------|
| **Tube** | 48 | 226,023 | 51 | 323,896 | **1.43x** | ani_dyn vs sph |
| **Tube** | 73 | 228,018 | 75 | 398,887 | **1.75x** | ani_dyn vs sph |
| **Tube** | 92 | 294,668 | 91 | 466,098 | **1.58x** | ani_dyn vs sph |
| **Tube** | 112 | 328,902 | 117 | 535,540 | **1.63x** | ani_dyn vs sph |
| **Tube** | 133 | 422,799 | 136 | 671,902 | **1.59x** | ani_dyn vs sph |
| **Ellipsoid** | 62 | 2,554,287 | 59 | 3,476,000 | **1.36x** | ani_dyn vs sani |
| **Ellipsoid** | 87 | 4,197,262 | 80 | 6,610,000 | **1.57x** | ani_dyn vs sani |
| **Ellipsoid** | 110 | 6,358,683 | 120 | 13,000,000 | **2.05x** | ani_dyn vs sani |
| **Ellipsoid** | 136 | 9,391,800 | 138 | 14,100,000 | **1.50x** | ani_dyn vs sani |
| **Ellipsoid** | 171 | 12,204,627 | 178 | 20,900,000 | **1.71x** | ani_dyn vs sani |
| **Plate** | 97 | 717,884 | 91 | 247,000 | **0.29x** | ani_dyn vs sph |
| **Plate** | 137 | 1,001,269 | 128 | 387,000 | **0.26x** | ani_dyn vs sph |
| **Plate** | 168 | 1,442,932 | 165 | 521,000 | **0.23x** | ani_dyn vs sph |
| **Plate** | 208 | 1,940,721 | 204 | 841,000 | **0.23x** | ani_dyn vs sph |
| **Plate** | 255 | 2,746,749 | 254 | 1,370,000 | **0.20x** | ani_dyn vs sph |

### Time Summary

| Mesh | Static (280x280) | Dynamic (280x280) | Default-Static (280x280) |
|------|------------------|-------------------|--------------------------|
| **Tube** | ani **2.0-3.7x faster** | sph 0.62-0.78x faster | ani_dyn **1.4-1.8x faster** vs sph |
| **Ellipsoid** | ani **1.3-1.6x faster** | N/A | ani_dyn **1.4-2.0x faster** vs sani |
| **Plate** | sph 0.80-0.90x slower | sph **4-6x faster** | sph **3-5x faster** vs ani_dyn |
