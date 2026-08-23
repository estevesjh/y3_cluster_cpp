# Redshift-contracted one-halo shear

This strategy evaluates the exact fixed-GL redshift contraction first and then
performs the mass sum against the production one-halo profile. It is the
maintained fast path for one-halo shear.

## Numerical definition

At fixed mass, define

$$
S_{ij}(M,z)=S_j(z)\int d\lambda_{\rm tr}
S_i(\lambda_{\rm tr},z)P_{\rm HOD}(\lambda_{\rm tr}\mid M,z).
$$

The lensing weight is

$$
W_{ij}(M)=\int dz\;n(M,z)\frac{dV}{d\Omega dz}\Omega(z)
\Sigma_{\rm crit}^{-1}(z)S_{ij}(M,z),
$$

and the one-halo result is

$$
O_{ij}(R)=\int d\ln M\;W_{ij}(M)
\left[(1-f_{\rm mis})\Delta\Sigma_{\rm cen}(R,M)
 +f_{\rm mis}\Delta\Sigma_{\rm mis}(R,M)\right].
$$

The production fast path evaluates or interpolates \(S_{ij}\) on its fixed
grid, then performs this mass contraction directly on the two
\(\Delta\Sigma\) components.

## Common algorithm

1. Build the fixed GL mass/redshift nodes and read the production selection
   table.
2. Interpolate the selection factor at each node using the production
   bilinear/clamp convention.
3. Multiply by the HMF, volume element, survey area, inverse critical density,
   and the richness/redshift weights.
4. Contract over redshift to obtain \(W_{ij}(M)\).
5. Evaluate the production miscentred mixture at every requested radius and
   contract over mass.

For the traditional max model, the redshift contraction remains explicit in
the final sum because (Delta\Sigma_{hh}(R,z)) depends on redshift:

$$
O_{ij}^{\rm max}(R)=\int dz\,d\ln M\;W_{ij}(M,z)
\Sigma_{\rm crit}^{-1}(z)\Delta\Sigma_{\rm max}(R,M,z).
$$

## Language implementations

| Language | Algorithm and source files | Status |
| --- | --- | --- |
| Python | `python/shear1h_fast_mass.py` uses `shared/datablock_models.py` and `shared/lensing_profiles.py`; `python/shear1h2h_max.py` is the z-resolved max model. | Readable replicas |
| C++ | `cpp/Shear1hFastMass.cc` is the one-halo identity wrapper; `cpp/Shear1h2hMax.cc` is the traditional max model. Both compose immutable GL cores. | Maintained CPU backends |
| CUDA | `cuda/Shear1h2hMaxGpu.cu` contracts the heavy miscentred-NFW part of the max model on the device; host code supplies the remaining tables and weights. | Max model only |

## Precision and cost

Pinned 12-bin × 10-radius fiducial measurements:

| Backend | Cost | Comparison |
| --- | ---: | --- |
| One-halo Python | 74 ms/sample | (8.4\times10^{-4}) vs adaptive full-ltmz; (3.1\times10^{-15}) vs production |
| One-halo C++ | 9 ms/sample | (8.4\times10^{-4}) vs adaptive; production identity |
| Max model C++ | 11 ms/sample | (8.3\times10^{-4}) vs adaptive; (6.0\times10^{-15}) vs Python |
| Max model CUDA / A100 | 8 ms/sample | (6.4\times10^{-15}) vs C++ twin |

The max-model two-halo term has a separate known numerical/debugging issue;
see `docs/known_issues/dsigma_hh_debug_flag.md` and the validation notes before using it as
a scientific reference.
