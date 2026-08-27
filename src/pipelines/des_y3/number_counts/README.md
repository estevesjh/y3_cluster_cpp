# Cluster number counts

This observable predicts the expected number of clusters in each observed
richness and observed-redshift bin. There is no radial profile. The
integrand composition follows the DES cluster-cosmology backbone
(DES Cluster et al. 2023) with the Costanzi et al. (2026) richness and
selection kernels.

## The physics

For richness bin $i$ and observed-redshift bin $j$,

$$
N_{ij} = \int dz\,d\ln M\,d\lambda_{\rm tr}\;
n(M,z)\frac{dV}{d\Omega dz}\Omega(z)
S_j(z)\,S_i(\lambda_{\rm tr},z)\,
P_{\rm HOD}(\lambda_{\rm tr}\mid M,z),
$$

where $n(M,z)$ is the halo mass function, $dV/d\Omega dz$ the comoving
volume element, $\Omega(z)$ the survey area, $P_{\rm HOD}$ the
shifted-Poisson true-richness distribution, $S_i$ the observed-richness
kernel (EMG projection model, Costanzi et al. 2026), and $S_j$ the
Gaussian observed-redshift kernel.

The dims tag of each backend states how many of the three integration
dimensions are handled **adaptively** — the cost driver:

- **`3d`** — all three dimensions adaptive (Cuhre on CPU, PAGANI on
  GPU). No approximation beyond the requested tolerance; this is the
  reference every other backend is quoted against.
- **`0d`** — no adaptive integration. Either the full explicit
  composition on fixed Gauss–Legendre grids (exact node placement makes
  this safe; see [GL nodes and weights](#gl-nodes-and-weights)), or the
  production fast path, which additionally replaces the
  $(\lambda_{\rm tr})$ integral and both kernels by the tabulated
  selection factor $S_{ij}(\ln M, z)$ and contracts the $z$ integral
  once per sample. The physical justification for the contraction is
  that no factor of the count integrand couples $z$ to a radial
  operator — the whole $z$ dependence can be summed into a mass-only
  weight $W_{ij}(M)$ before the mass integral.

## Precision and cost

Pinned 12-bin fiducial measurements; cost is per sample. Precision is
quoted against the `3d` adaptive reference; where a number was measured
against a different baseline, the baseline is stated.

| Dims | Method and backend | Cost | Precision vs 3d |
| --- | --- | ---: | --- |
| `3d` | [adaptive Python mass reference](#the-3d-backends) (`shared/explicit_grid_core.py`) | 25 s | Reference (3d); reported integration error at or below 1e-6 |
| `3d` | [Cuhre C++](#the-3d-backends) | 3.1 s | 4.9e-4 (baseline: the 0d explicit fixed-GL Python, which is 3.5e-5 from the 3d reference) |
| `3d` | [PAGANI CUDA/A100](#the-3d-backends) | 2.0 s | 5.1e-4 (same fixed-GL baseline) |
| `0d` | [explicit Python (3-dim GL)](#explicit-fixed-gl-python-reference) | 83 ms | 3.5e-5 |
| `0d` | [fast path Python (2-dim GL, S_ij tab)](#redshift-contracted-fast-path) | 5 ms | 7.6e-4; also 2.4e-15 vs production (separate baseline) |
| `0d` | [fast path C++ (2-dim GL, S_ij tab)](#redshift-contracted-fast-path) | 6 ms | 7.6e-4; also identity with production (separate baseline) |

## The 0d backends

Every backend here has **zero adaptive integration dimensions** — all
integrals are fixed Gauss–Legendre sums, so everything is fast and
MCMC-viable. Two algorithms share the folder.

### Redshift-contracted fast path

(Formerly `fast_mass`.) Reproduces the production count calculation by
separating the redshift and selection contraction from the remaining
mass integral. Define the selection factor at fixed mass and redshift,

$$
S_{ij}(M,z)=S_j(z)\int d\lambda_{\rm tr}\,
S_i(\lambda_{\rm tr},z)\,P_{\rm HOD}(\lambda_{\rm tr}\mid M,z),
$$

which the production path tabulates on a fixed $(\ln M,z)$ grid and
bilinearly interpolates. The redshift contraction is then

$$
W_{ij}(M)=\int dz\;n(M,z)\frac{dV}{d\Omega dz}\Omega(z)S_{ij}(M,z),
\qquad
N_{ij}=\int d\ln M\;W_{ij}(M).
$$

The radial-series population moments use the same $W_{ij}(M)$, which is
why the fixed-GL weight builder is kept in the shared layer: the
pipeline-owned `shared/sel_gl_weights.hh`
(`y3_pipelines::SelGlWeights`, identity-certified against the
production `SelGLCore`) on the C++ side, and
`shared/datablock_models.py::MassZWeights` on the Python side. This is a
computational re-expression of the production algorithm: it does not
remove the production table's interpolation error, and the residual
against the adaptive reference (7.6e-4) is dominated by that
tabulation, not by any difference between the Python and C++
implementations (which agree with production to 2.4e-15 / identity).

### Explicit fixed-GL Python reference

(Formerly the fixed-GL Python half of `full_ltmz`.) Evaluates the full
explicit $(\lambda_{\rm tr}, \ln M, z)$ composition — selection kernels
at the quadrature nodes, no $S_{ij}$ tabulation — on fixed GL grids
(96 mass, 64 redshift, 32 true-richness nodes), so it carries zero
adaptive dimensions. It sits 3.5e-5 from the adaptive mass reference
and is the certified stand-in the adaptive C++/CUDA backends are
compared against. Its adaptive twins are
[the 3d backends](#the-3d-backends).

### GL nodes and weights

All fixed grids use standard Gauss–Legendre quadrature: nodes and
weights on $[-1,1]$ affine-mapped to the integration interval,
$x' = \tfrac{b-a}{2}x + \tfrac{b+a}{2}$, $w' = \tfrac{b-a}{2}w$
(`shared/datablock_models.py::gl_nodes`, via
`numpy.polynomial.legendre.leggauss`; the C++ twin
`y3_pipelines::gl_nodes` in the pipeline-owned
`shared/sel_gl_weights.hh` computes the same Legendre roots by Newton
iteration and applies the same map). The mass grid places `n_lnm` nodes (default 96) on
$[\ln M_{\rm low}, \ln M_{\rm high}]$ and the redshift grid `n_z` nodes
(default 64) on $[z_{\rm low}, z_{\rm high}]$; the composed weight is
$W(b;k,q) = w^z_q\,\tfrac{dV}{d\Omega dz}(z_q)\,\Omega(z_q)\,
n(\ln M_k, z_q)\,S(b;\ln M_k,z_q)$, and the observable is the doubly
weighted sum $\sum_k w^{\ln M}_k \sum_q W$.

The true-richness integral is where node placement carries the
accuracy: the shifted-Poisson $P_{\rm HOD}$ is a near-delta ridge at
low mass, so `n_q` GL nodes (default 32) are placed **per
$(\ln M, z)$ node** on the HOD support
$[\max(0,\mu_{\rm eff}-L_\lambda\sigma_{\rm eff}),\;
\mu_{\rm eff}+L_\lambda\sigma_{\rm eff}]$ with
$\mu_{\rm eff}=\lambda_{\rm cen}+\mu_{\rm sat}$,
$\sigma_{\rm eff}=\sqrt{\mu_{\rm sat}+(\sigma_\lambda\mu_{\rm sat})^2}$
and $L_\lambda$ = `l_lam` (default 6)
(`shared/sel_function.py::_compute_lam_nodes_and_P_HOD`). Hand-placing
the panel on the integrand feature is exactly why these fixed grids
reach 3.5e-5 while a global adaptive volume must spend seconds finding
the same ridge. Convergence is certified by doubled-node
self-convergence (96→192 mass, 64→128 redshift, 32→64 richness nodes
and $L_\lambda$ 6→8) in
`src/pipelines/des_y3/validate_against_fiducial.py`.

## The 3d backends

(Formerly `full_ltmz`; three adaptive dimensions.) The independent
reference: the same explicit integrand $F_{ij}(\lambda_{\rm tr},M,z)$
as above, with the true-richness limits
$\lambda_\pm=\mu_{\rm eff}\pm L_\lambda\sigma_{\rm eff}$ (lower bound
clipped at zero), integrated by adaptive Cuhre (C++) or PAGANI (CUDA)
over the full three-dimensional volume per bin. The adaptive Python
mass reference in `shared/explicit_grid_core.py` uses the same inner
fixed-GL contraction and adaptively subdivides the outer mass integral
to a reported error at or below 1e-6.

`eps_rel=1e-3` is not accepted for the Cuhre reference: it can
under-integrate the near-delta richness ridge by about one percent in
the lowest-richness bin; the accepted setting is `eps_rel=1e-4`.

The CUDA backend composes the gpu_prj_costanzi2026 device models
(`models/mor_shifted_poisson_t.cuh`, `models/emg_des_t.cuh`; Arwa Qadi,
upstream PR #3) plus the local `zkernel_sj`. Note the MOR convention
offset documented in the header (Costanzi-2026 form, no central-count
shift; `MOR_SP(ltr) = MOR_HOD(ltr+1)` above `Mmin`).

## Running the backends

| Dims | Language | Sources | Module / output |
| --- | --- | --- | --- |
| `0d` | Python | `python/0d/numcounts_sij_gl.py` (+ validator `python/0d/validate_fast_vs_production.py`) | `numcounts_sij_gl/vals` |
| `0d` | C++ | `cpp/0d/NumCountsSijGl.cc`, physics in `cpp/0d/num_counts_sij_gl_t.hh` | `NumCountsSijGl.so` in `release-build/src/modules/des_y3_numcounts_0d_cpp/`, section `numcounts_sij_gl` |
| `0d` | Python | `python/0d/numcounts_explicit_gl.py` (+ validator `python/0d/validate_explicit_vs_production.py`) | `numcounts_explicit_gl/vals` |
| `3d` | C++ | `cpp/3d/NumCounts3d.cc`, physics in `cpp/3d/num_counts_3d_t.hh` | `NumCounts3d.so` in `des_y3_numcounts_3d_cpp/`, `numcounts3d/{vals,errors,probs,status,nregions}` |
| `3d` | CUDA | `cuda/3d/NumCounts3dGpu.cu`, physics in `cuda/3d/num_counts_3d_gpu_t.cuh` | `NumCounts3dGpu.so` in `des_y3_numcounts_3d_cuda/`, `numcounts3dgpu/{...}` |

No CUDA `0d` backend is provided: a one-dimensional mass contraction
does not justify a GPU kernel in the current pipeline. Module labels
are the ini `[section]` names, so a pipeline drives these backends with
`[NumCountsSijGl]`-style sections pointing at the `.so` paths above.
