# Explicit one-halo shear integration

This strategy is the independent reference for the maintained one-halo shear
observable. It keeps true richness, true redshift, and halo mass explicit and
evaluates the selection kernels directly at quadrature nodes.

## Numerical definition

For each richness/redshift bin and projected radius,

$$
O_{ij}(R)=\int dz\,d\ln M\,d\lambda_{\rm tr}\;
n(M,z)\frac{dV}{d\Omega dz}\Omega(z)
\Sigma_{\rm crit}^{-1}(z)
S_j(z)S_i(\lambda_{\rm tr},z)
P_{\rm HOD}(\lambda_{\rm tr}\mid M,z)
\left[
(1-f_{\rm mis})\Delta\Sigma_{\rm cen}(R,M,z)
 +f_{\rm mis}\Delta\Sigma_{\rm mis}(R,M,z)
\right].
$$

The fixed-GL implementation uses 96 mass nodes, 64 redshift nodes, and 32
true-richness nodes. The adaptive reference shares the inner fixed-GL
selection contraction and adaptively subdivides the outer mass integral.

## Common algorithm

1. Load the selection kernels, HMF, volume element, inverse critical density,
   and the production centred/miscentred profile adapters.
2. Place nodes in \(z\), \(\ln M\), and the per-\((M,z)\) true-richness
   bracket.
3. Evaluate the selection weight, multiply by
   \(\Sigma_{\rm crit}^{-1}\left[(1-f_{\rm mis})\Delta\Sigma_{\rm cen}
   +f_{\rm mis}\Delta\Sigma_{\rm mis}\right]\), and apply all quadrature
   weights.
4. Sum over the three integration variables for every bin and radius.
5. For the adaptive reference, split the outer mass interval until the
   embedded error estimate meets `eps_rel`.

## Language implementations

| Language | Algorithm and source files | Output |
| --- | --- | --- |
| Python | `python/shear1h_full_ltmz.py` uses `shared/full_ltmz_core.py` and `shared/lensing_profiles.py`. | `shear1h_full_ltmz/vals` |
| C++ | `cpp/Shear1hFullLtmz.cc` and its immutable integration template evaluate one adaptive Cuhre integral per bin/radius point. | `shear1hfullltmz/{vals,errors,probs,status,nregions}` |
| CUDA | `cuda/Shear1hFullLtmzGpu.cu` reuses the count device kernels and evaluates the profile-weighted integral with PAGANI. | GPU module output section |

The language implementations share the physical profile and kernel
conventions. They differ only in quadrature and execution language.

## Precision and cost

Pinned 12-bin × 10-radius fiducial measurements:

| Backend | Cost | Comparison |
| --- | ---: | --- |
| Python adaptive reference | 35 s/sample | Fiducial reference; reported error \(\le 10^{-6}\) |
| Python fixed GL | 149 ms/sample | \(4.9\times10^{-5}\) vs adaptive |
| C++ Cuhre, `eps_rel=1e-4` | 51 s/sample | \(3.3\times10^{-4}\) vs Python reference |
| CUDA PAGANI, A100 | 32 s/sample | \(3.4\times10^{-4}\) vs C++ reference |

The full-ltmz implementations are reference tools. The maintained production
timing is supplied by the fast-mass C++ backend.
