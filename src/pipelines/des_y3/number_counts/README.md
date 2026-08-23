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

| Method and backend | Cost | Comparison or status |
| --- | ---: | --- |
| [`full_ltmz`](full_ltmz/README.md), adaptive Python | 25 s | Reference; reported integration error at or below 1e-6 |
| [`full_ltmz`](full_ltmz/README.md), fixed GL Python | 83 ms | 3.5e-5 vs adaptive reference |
| [`full_ltmz`](full_ltmz/README.md), Cuhre C++ | 3.1 s | 4.9e-4 vs fixed-GL reference |
| [`full_ltmz`](full_ltmz/README.md), PAGANI CUDA/A100 | 2.0 s | 5.1e-4 vs fixed-GL reference |
| [`fast_mass`](fast_mass/README.md), Python | 5 ms | 7.6e-4 vs adaptive; identity with production |
| [`fast_mass`](fast_mass/README.md), C++ | 6 ms | 7.6e-4 vs adaptive; identity with production |

The detailed strategy README contains the grids, tolerances, implementation
files, and validation procedure.
