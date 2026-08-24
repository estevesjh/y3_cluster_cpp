# Explicit adaptive count integration (`3d`)

This strategy (formerly `full_ltmz`; three adaptive dimensions) is the
independent reference for cluster counts. It evaluates the selection kernels
at the quadrature nodes instead of using the production selection table
$S_{ij}(\ln M,z)$. The fixed-GL Python twin of the same explicit composition
carries zero adaptive dimensions and lives in
[`../0d/`](../0d/README.md#explicit-fixed-gl-python-reference).

## Numerical definition

The calculation is

$$
N_{ij} = \int_{z_{\min}}^{z_{\max}} dz
\int_{\ln M_{\min}}^{\ln M_{\max}} d\ln M
\int_{\lambda_-(M,z)}^{\lambda_+(M,z)} d\lambda_{\rm tr}\,
F_{ij}(\lambda_{\rm tr},M,z),
$$

with

$$
F_{ij}=n(M,z)\frac{dV}{d\Omega dz}\Omega(z)
S_j(z)S_i(\lambda_{\rm tr},z)
P_{\rm HOD}(\lambda_{\rm tr}\mid M,z).
$$

The true-richness limits are the per-$(M,z)$ bracket used by the maintained
richness model,

$$
\lambda_\pm=\mu_{\rm eff}(M,z)\pm L_\lambda\sigma_{\rm eff}(M,z),
$$

with the lower bound clipped at zero. The default fixed-GL grid is 96 mass
nodes, 64 redshift nodes, and 32 true-richness nodes. The adaptive reference
uses the same inner fixed-GL contraction and adaptively subdivides the outer
mass integral.

## Common algorithm

1. Read the halo mass function, volume element, survey area, HOD, richness
   kernel, photo-$z$ kernel, and bin limits.
2. Place fixed Gauss--Legendre nodes in $z$ and $\ln M$.
3. At every $(M,z)$, construct the true-richness bracket and place the
   true-richness nodes inside it.
4. Evaluate the HOD and both observed-bin kernels at those nodes.
5. Multiply by $n\,dV/d\Omega dz\,\Omega$ and sum with the three sets of
   quadrature weights.
6. Return one expected count per configured bin, together with quadrature
   diagnostics in the adaptive backends.

## Language implementations

| Language | Algorithm and source files | Output |
| --- | --- | --- |
| Python | No module here: the fixed-GL Python twin (`numcounts_full_ltmz.py`) has zero adaptive dimensions and lives in [`../0d/`](../0d/README.md#explicit-fixed-gl-python-reference). The shared adaptive mass reference is in `shared/full_ltmz_core.py`. | see `../0d/` |
| C++ | `cpp/NumCountsFullLtmz.cc` instantiates the immutable scalar integration template; `cpp/num_counts_full_ltmz_t.hh` supplies the Cuhre integrand and model composition. | `numcountsfullltmz/{vals,errors,probs,status,nregions}` |
| CUDA | `cuda/NumCountsFullLtmzGpu.cu` instantiates the CUDA integration macro; `cuda/num_counts_full_ltmz_gpu_t.cuh` supplies the PAGANI integrand, composing the gpu_prj_costanzi2026 device models (`models/mor_shifted_poisson_t.cuh`, `models/emg_des_t.cuh`; Arwa Qadi, upstream PR #3) plus the local `zkernel_sj`. Note the MOR convention offset documented in the header (Costanzi-2026 form, no central-count shift). | `numcountsfullltmzgpu/{vals,errors,probs,status,nregions}` |

All three evaluate the same mathematical integral. They differ in how the
outer integration is performed: fixed GL in Python, adaptive Cuhre on CPU,
and adaptive PAGANI on GPU.

## Precision and cost

Pinned 12-bin fiducial measurements:

| Dims | Backend | Cost | Precision vs 3d |
| --- | --- | ---: | --- |
| `3d` | Python adaptive mass reference | 25 s/sample | Reference (3d); reported error $\le 10^{-6}$ |
| `0d` | Python fixed GL (3-dim GL, in `../0d/`) | 83 ms/sample | $3.5\times10^{-5}$ |
| `3d` | C++ Cuhre, `eps_rel=1e-4` | 3.1 s/sample | $4.9\times10^{-4}$ (baseline: the 0d fixed-GL Python, itself $3.5\times10^{-5}$ from the 3d reference) |
| `3d` | CUDA PAGANI, A100 | 2.0 s/sample | $5.1\times10^{-4}$ (same fixed-GL baseline) |

`eps_rel=1e-3` is not accepted for the Cuhre reference: it can under-integrate
the near-delta richness ridge by about one percent in the lowest-richness bin.
