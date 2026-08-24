# One-halo and traditional one-plus-two-halo shear

This observable contains the maintained miscentered one-halo model and the
optional traditional max model.

## Numerical definitions

The one-halo profile is a centered/miscentered mixture,

$$
\Delta\Sigma_{1h}(R,M,z)=
(1-f_{\rm mis})\Delta\Sigma_{\rm cen}(R,M,z)
+f_{\rm mis}\Delta\Sigma_{\rm mis}(R,M,z).
$$

The stacked excess surface density is

$$
O_{ij}(R)=\int dz\,d\ln M\,d\lambda_{\rm tr}\;
n(M,z)\frac{dV}{d\Omega dz}\Omega(z)
S_j(z)S_i(\lambda_{\rm tr},z)
P_{\rm HOD}(\lambda_{\rm tr}\mid M,z)
\Delta\Sigma_{1h}(R,M,z).
$$

The `Sigma_crit^{-1}` factor is optional. Set it to unity when computing
`DeltaSigma`; including the lensing-geometry factor instead returns the
tangential shear `gamma_T`.

The optional traditional max model retains the redshift-dependent two-halo
term,

$$
\Delta\Sigma_{\rm max}(R,M,z)=
\max\left[\Delta\Sigma_{1h}(R,M,z),
b(M,z)\Delta\Sigma_{hh}(R,z)\right].
$$

The numerical implementations are documented in the strategy READMEs,
named by the number of adaptive integration dimensions:
[`3d`](3d/README.md) (formerly `full_ltmz`, the adaptive explicit
references) and [`0d`](0d/README.md) (no adaptive integration — the
fixed-GL and tabulated fast backends: the one-halo z-contracted sum and
the max model's z-resolved sum, both formerly `fast_mass`; the explicit
fixed-GL Python references; and the moment-expanded radial series,
formerly `radial_series`).

## Precision and cost

Pinned 12-bin by 10-radius fiducial measurements; cost is per sample.

| Dims | Method and backend | Cost | Precision vs 3d |
| --- | --- | ---: | --- |
| `3d` | [`3d`](3d/README.md), adaptive Python | 35 s | Reference (3d); reported integration error at or below 1e-6 |
| `3d` | [`3d`](3d/README.md), Cuhre C++ | 51 s | 3.3e-4 vs the 3d Python reference |
| `3d` | [`3d`](3d/README.md), PAGANI CUDA/A100 | 32 s | 3.4e-4 (baseline: the 3d C++ twin) |
| `0d` | [`0d`](0d/README.md), explicit Python (3-dim GL) | 149 ms | 4.9e-5 |
| `0d` | [`0d`](0d/README.md), one-halo Python/C++ (1-dim GL, z contracted) | 74 / 9 ms | 8.4e-4; also identity with production (separate baseline) |
| `0d` | [`0d`](0d/README.md), max C++/CUDA/A100 (2-dim GL, z-resolved) | 11 / 8 ms | 8.3e-4; CUDA agrees with C++ to 6.4e-15 (twin baseline) |
| `0d` | [`0d`](0d/README.md), radial series Python/C++ (tables + moments) | 6--7 ms | 56--86% vs 3d (known fixed-c=4 defect, [docs/known_issues](../../../../docs/known_issues/radial_series_vs_full_ltmz_defect.md)); 3.7e-3 internal fixed-profile consistency |

The traditional max model is provisional because its two-halo input has a
separate `haloModel/dSigma_hh` data defect.

All backends read their lensing profile tables (`dSigma_nfw`, `dSigma_hh`,
`bias`, `concentration`) from the `haloModel` datablock section, whose
halo-model physics lives in the cosmology layer:
[`../../cosmology/halo_model.py`](../../cosmology/halo_model.py) (driven
through the CosmoSIS wrapper `y3_buzzard/halo_model_cosmosis.py`).
