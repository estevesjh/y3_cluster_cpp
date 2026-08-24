# Explicit adaptive one-halo shear integration (`3d`)

This strategy (formerly `full_ltmz`; three adaptive dimensions) is the
independent reference for the maintained one-halo shear observable. It keeps
true richness, true redshift, and halo mass explicit and evaluates the
selection kernels directly at quadrature nodes. The fixed-GL Python twins of
the same explicit composition carry zero adaptive dimensions and live in
[`../0d/`](../0d/README.md#explicit-fixed-gl-python-references).

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
| Python | No module here: the fixed-GL twin (`shear1h_full_ltmz.py`) has zero adaptive dimensions and lives in [`../0d/`](../0d/README.md#explicit-fixed-gl-python-references). | see `../0d/` |
| C++ | `cpp/Shear1hFullLtmz.cc` (thin driver) with the physics in `cpp/shear1h_full_ltmz_t.hh`; one adaptive Cuhre integral per bin/radius point through the immutable integration template. | `shear1hfullltmz/{vals,errors,probs,status,nregions}` |
| CUDA | `cuda/Shear1hFullLtmzGpu.cu` (thin driver) with the physics in `cuda/shear1h_full_ltmz_gpu_t.cuh`, composing the gpu_prj_costanzi2026 device models; PAGANI quadrature. | `shear1hfullltmzgpu/{vals,errors,probs,status,nregions}` |

### Traditional 1h+2h max model, explicit references

The z-dependent max-model observable also has an explicit adaptive `3d`
reference backend here (the reference the `0d` `Shear1h2hMax` backends
validate against). Because the two-halo term is z-dependent and the max
is nonlinear, nothing contracts past the profile:

| Language | Algorithm and source files | Output |
| --- | --- | --- |
| Python | No module here: the fixed-GL twin (`shear1h2h_max_full_ltmz.py`) lives in [`../0d/`](../0d/README.md#explicit-fixed-gl-python-references). | see `../0d/` |
| C++ | `cpp/Shear1h2hMaxFullLtmz.cc` (thin driver) with the physics in `cpp/shear1h2h_max_full_ltmz_t.hh`; one adaptive Cuhre triple integral per (bin, R) with `d_tot = max(1h mixture, b · dSigma_hh)` inside the volume. | `shear1h2hmaxfullltmz/{vals,errors,probs,status,nregions}` |

Both require `compute_lensing_2h = T` and sanitize the `dSigma_hh` NaNs
to 0 before interpolation (exact under the max; see
`docs/known_issues/dsigma_hh_debug_flag.md`), and both read
`miscentering/{f_mis,tau_mis}` strictly — no in-code default fallback.

The language implementations share the physical profile and kernel
conventions. They differ only in quadrature and execution language.

## Precision and cost

Pinned 12-bin × 10-radius fiducial measurements:

| Dims | Backend | Cost | Precision vs 3d |
| --- | --- | ---: | --- |
| `3d` | Python adaptive reference (`shared/full_ltmz_core.py`) | 35 s/sample | Reference (3d); reported error \(\le 10^{-6}\) |
| `0d` | Python fixed GL (3-dim GL, in `../0d/`) | 149 ms/sample | \(4.9\times10^{-5}\) |
| `3d` | C++ Cuhre, `eps_rel=1e-4` | 51 s/sample | \(3.3\times10^{-4}\) vs the 3d Python reference |
| `3d` | CUDA PAGANI, A100 | 32 s/sample | \(3.4\times10^{-4}\) (baseline: the 3d C++ twin) |

These explicit implementations are reference tools. The maintained production
timing is supplied by the `0d` fixed-GL C++ backends.
