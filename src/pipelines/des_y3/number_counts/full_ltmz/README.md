# Explicit count integration

This strategy is the independent reference for cluster counts. It evaluates
the selection kernels at the quadrature nodes instead of using the production
selection table $S_{ij}(\ln M,z)$.

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
| Python | Fixed-GL module in `python/numcounts_full_ltmz.py`; the shared adaptive mass reference is in `shared/full_ltmz_core.py`. Kernels are imported from `shared/sel_kernels.py`, not copied. | `numcounts_full_ltmz/vals` |
| C++ | `cpp/NumCountsFullLtmz.cc` instantiates the immutable scalar integration template; `cpp/num_counts_full_ltmz_t.hh` supplies the Cuhre integrand and model composition. | `numcountsfullltmz/{vals,errors,probs,status,nregions}` |
| CUDA | `cuda/NumCountsFullLtmzGpu.cu` uses PAGANI. Device kernel ports live in `cuda/full_ltmz_device_kernels.cuh`; fixed-size integrand state is copied to the device. | `numcountsfullltmzgpu/{vals,errors,probs,status,nregions}` |

All three evaluate the same mathematical integral. They differ in how the
outer integration is performed: fixed GL in Python, adaptive Cuhre on CPU,
and adaptive PAGANI on GPU.

## Precision and cost

Pinned 12-bin fiducial measurements:

| Backend | Cost | Comparison |
| --- | ---: | --- |
| Python adaptive mass reference | 25 s/sample | Reference; reported error $\le 10^{-6}$ |
| Python fixed GL | 83 ms/sample | $3.5\times10^{-5}$ vs adaptive |
| C++ Cuhre, `eps_rel=1e-4` | 3.1 s/sample | $4.9\times10^{-4}$ vs fixed-GL reference |
| CUDA PAGANI, A100 | 2.0 s/sample | $5.1\times10^{-4}$ vs fixed-GL reference |

`eps_rel=1e-3` is not accepted for the Cuhre reference: it can under-integrate
the near-delta richness ridge by about one percent in the lowest-richness bin.
