# Cluster number counts

This observable predicts the expected number of clusters in each observed
richness and observed-redshift bin. There is no radial profile.

## Numerical definition

For richness bin `i` and observed-redshift bin `j`,

$$
N_{ij} = \int dz\,d\ln M\,d\lambda_{\rm tr}\;
n(M,z)\frac{dV}{d\Omega dz}\Omega(z)
S_j(z)S_i(\lambda_{\rm tr},z)
P_{\rm HOD}(\lambda_{\rm tr}\mid M,z).
$$

Here `n(M,z)` is the halo mass function, `P_HOD` is the true-richness
distribution, `S_i` maps true richness to observed richness, and `S_j` is the
photometric-redshift selection.

## Precision and cost

Pinned 12-bin fiducial measurements; cost is per sample.

Strategy folders count adaptive integration dimensions: [`3d`](3d/README.md)
(formerly `full_ltmz`) holds the adaptive explicit C++/CUDA references;
[`0d`](0d/README.md) holds everything with no adaptive integration — the
redshift-contracted fast path (formerly `fast_mass`) and the explicit
fixed-GL Python reference.

| Dims | Method and backend | Cost | Precision vs 3d |
| --- | --- | ---: | --- |
| `3d` | [`3d`](3d/README.md), adaptive Python (`shared/full_ltmz_core.py`) | 25 s | Reference (3d); reported integration error at or below 1e-6 |
| `3d` | [`3d`](3d/README.md), Cuhre C++ | 3.1 s | 4.9e-4 (baseline: the 0d explicit fixed-GL Python, which is 3.5e-5 from the 3d reference) |
| `3d` | [`3d`](3d/README.md), PAGANI CUDA/A100 | 2.0 s | 5.1e-4 (same fixed-GL baseline) |
| `0d` | [`0d`](0d/README.md), explicit Python (3-dim GL) | 83 ms | 3.5e-5 |
| `0d` | [`0d`](0d/README.md), fast path Python (2-dim GL, S_ij tab) | 5 ms | 7.6e-4; also identity with production (separate baseline) |
| `0d` | [`0d`](0d/README.md), fast path C++ (2-dim GL, S_ij tab) | 6 ms | 7.6e-4; also identity with production (separate baseline) |

The detailed strategy README contains the grids, tolerances, implementation
files, and validation procedure.
