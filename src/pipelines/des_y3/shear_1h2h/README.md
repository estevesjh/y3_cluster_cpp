# One-halo and traditional one-plus-two-halo shear

This observable contains the maintained miscentered one-halo model and
the optional traditional max model, on the DES cluster-cosmology
backbone (DES Cluster et al. 2023) with the Costanzi et al. (2026)
richness and selection kernels.

## The physics

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
S_j(z)\,S_i(\lambda_{\rm tr},z)\,
P_{\rm HOD}(\lambda_{\rm tr}\mid M,z)\,
\Delta\Sigma_{1h}(R,M,z),
$$

with $S_i$ the observed-richness kernel, $S_j$ the observed-redshift
kernel, and $P_{\rm HOD}$ the shifted-Poisson true-richness
distribution. The $\Sigma_{\rm crit}^{-1}$ factor is optional: set it to
unity to compute $\Delta\Sigma$; include the lensing-geometry factor to
return the tangential shear $\gamma_T$. The optional traditional max
model retains the redshift-dependent two-halo term,

$$
\Delta\Sigma_{\rm max}(R,M,z)=
\max\left[\Delta\Sigma_{1h}(R,M,z),\;
b(M,z)\Delta\Sigma_{hh}(R,z)\right].
$$

The dims tag of each backend counts its **adaptive** integration
dimensions, and the physics dictates which reductions are legal:

- **`3d`** — all three dimensions adaptive; the reference class.
- **`0d`** — no adaptive integration. The one-halo profile at fixed
  concentration is $z$-independent, so the $z$ (and $\lambda_{\rm tr}$)
  integrals commute past it and can be contracted once per sample into
  a mass-only weight — a single fixed-GL mass sum remains. The max
  model's two-halo term **is** $z$-dependent and the max is nonlinear,
  so nothing commutes: its `0d` backend keeps the $z$-resolved weight
  and performs a double $(\ln M, z)$ GL sum. The explicit fixed-GL
  Python references evaluate the full composition on fixed grids with
  feature-placed nodes; the radial series replaces runtime integration
  entirely with offline profile tables and population moments.

## Precision and cost

Pinned 12-bin by 10-radius fiducial measurements; cost is per sample.
Precision is quoted against the `3d` adaptive reference; other
baselines are stated in-cell.

| Dims | Method and backend | Cost | Precision vs 3d |
| --- | --- | ---: | --- |
| `3d` | [adaptive Python reference](#the-3d-backends) | 35 s | Reference (3d); reported integration error at or below 1e-6 |
| `3d` | [Cuhre C++](#the-3d-backends) | 51 s | 2.6e-4 direct vs the 3d Python reference (re-measured 2026-08-26 on the reduced 6-pt corner wall, same fiducial point) |
| `3d` | [PAGANI CUDA/A100](#the-3d-backends) | 32 s | 3.0e-4 direct vs the 3d Python reference (re-measured 2026-08-26); CUDA vs C++ twin 4.3e-5 (separate baseline, shared A100) |
| `0d` | [explicit Python (3-dim GL)](#explicit-fixed-gl-python-references) | 149 ms | 4.9e-5 |
| `0d` | [one-halo Python/C++ (1-dim GL, z contracted)](#one-halo-z-contracted-gl) | 74 / 9 ms | 8.4e-4; also identity with production (separate baseline) |
| `0d` | [max C++/CUDA/A100 (2-dim GL, z-resolved)](#traditional-1h2h-max-model-z-resolved-gl) | 11 / 8 ms | 8.3e-4; CUDA agrees with C++ to 6.4e-15 (twin baseline) |
| `0d` | [radial series Python/C++ (tables + moments)](#moment-expanded-radial-series) | 6--7 ms | 56--86% vs 3d (known fixed-c=4 defect, [docs/known_issues](../../../../docs/known_issues/radial_series_vs_full_ltmz_defect.md)); 3.7e-3 internal fixed-profile consistency |

The traditional max model is provisional because its two-halo input has
a separate `haloModel/dSigma_hh` data defect. All backends read their
lensing profile tables (`dSigma_nfw`, `dSigma_hh`, `bias`,
`concentration`) from the `haloModel` datablock section, whose
halo-model physics lives in the cosmology layer:
[`../../cosmology/halo_model.py`](../../cosmology/halo_model.py) (driven
through the CosmoSIS wrapper `y3_buzzard/halo_model_cosmosis.py`).

## The 0d backends

Every backend here has zero adaptive integration dimensions — fixed
Gauss–Legendre sums or offline table lookups — so all are fast and
MCMC-viable. Four algorithms:

| Algorithm | Backends |
| --- | --- |
| [1h z-contracted GL](#one-halo-z-contracted-gl) | Python, C++ |
| [Max model z-resolved GL](#traditional-1h2h-max-model-z-resolved-gl) | Python, C++, CUDA |
| [Explicit fixed-GL references](#explicit-fixed-gl-python-references) | Python |
| [Moment-expanded radial series](#moment-expanded-radial-series) | Python, C++ |

### One-halo z-contracted GL

(Formerly `fast_mass`, one-halo half.) Because the fixed-concentration
one-halo profile is $z$-free, the redshift and selection contraction is
performed first. At fixed mass,

$$
S_{ij}(M,z)=S_j(z)\int d\lambda_{\rm tr}\,
S_i(\lambda_{\rm tr},z)\,P_{\rm HOD}(\lambda_{\rm tr}\mid M,z),
$$

$$
W_{ij}(M)=\int dz\;n(M,z)\frac{dV}{d\Omega dz}\Omega(z)
\Sigma_{\rm crit}^{-1}(z)\,S_{ij}(M,z),
$$

$$
O_{ij}(R)=\int d\ln M\;W_{ij}(M)
\left[(1-f_{\rm mis})\Delta\Sigma_{\rm cen}(R,M)
 +f_{\rm mis}\Delta\Sigma_{\rm mis}(R,M)\right].
$$

The production fast path evaluates or interpolates $S_{ij}$ on its
fixed grid and performs this mass contraction directly on the two
$\Delta\Sigma$ components. It is the maintained fast path for one-halo
shear. The weight builder is the pipeline-owned
`shared/sel_gl_weights.hh` (`y3_pipelines::SelGlWeights`,
identity-certified against the production `SelGLCore` — pipeline C++
includes no production operator headers); the Python twin is
`shared/datablock_models.py::MassZWeights`.

| Dims | Backend | Cost | Precision vs 3d |
| --- | --- | ---: | --- |
| `0d` | One-halo Python (1-dim GL, z contracted) | 74 ms/sample | $8.4\times10^{-4}$; also $3.1\times10^{-15}$ vs production (separate baseline) |
| `0d` | One-halo C++ (1-dim GL, z contracted) | 9 ms/sample | $8.4\times10^{-4}$; also production identity (separate baseline) |

### Traditional 1h+2h max model, z-resolved GL

(Formerly the max-model half of `fast_mass`.) The biased two-halo term
$b(M,z)\,\Delta\Sigma_{hh}(R,z)$ is $z$-dependent and the max is
nonlinear, so the redshift integral cannot be contracted past the
profile. The backend keeps the $z$-resolved weight — a double fixed-GL
$(\ln M, z)$ sum per (bin, R), still zero adaptive dimensions:

$$
O_{ij}^{\rm max}(R)=\int dz\,d\ln M\;W_{ij}(M,z)\,
\Sigma_{\rm crit}^{-1}(z)\,\Delta\Sigma_{\rm max}(R,M,z),
$$

with $W_{ij}(M,z)$ the $z$-resolved selection weight (same term
composition as the one-halo GL core, without the $z$ sum) and
$\Delta\Sigma_{1h}$ the production miscentred mixture. The $S_{ij}$
tabulation is still the fast-path hallmark.

| Dims | Backend | Cost | Precision vs 3d |
| --- | --- | ---: | --- |
| `0d` | Max model C++ (2-dim GL, z-resolved) | 11 ms/sample | $8.3\times10^{-4}$; also $6.0\times10^{-15}$ vs Python twin (separate baseline) |
| `0d` | Max model CUDA / A100 (2-dim GL, z-resolved) | 8 ms/sample | $6.4\times10^{-15}$ vs C++ twin (separate baseline); vs-3d inherited through the C++ row |

The explicit references this algorithm validates against are the
[3d backends](#the-3d-backends) (`Shear1h2hMax3d` C++) and the
[explicit fixed-GL section](#explicit-fixed-gl-python-references)
(`shear1h2h_max_explicit_gl.py`). The max-model two-halo term has a
separate known data defect; see
[`docs/known_issues/dsigma_hh_debug_flag.md`](../../../../docs/known_issues/dsigma_hh_debug_flag.md)
before using it as a scientific reference.

### Explicit fixed-GL Python references

(Formerly the fixed-GL Python half of `full_ltmz`.) The full explicit
$(\lambda_{\rm tr}, \ln M, z)$ composition — every selection kernel at
the quadrature nodes, no $S_{ij}$ tabulation — on fixed GL grids
(96 mass, 64 redshift, 32 true-richness nodes): zero adaptive
dimensions. Their adaptive C++/CUDA twins are the
[3d backends](#the-3d-backends).

| Dims | Module | Observable | Output | Cost / accuracy |
| --- | --- | --- | --- | --- |
| `0d` | `python/0d/shear1h_explicit_gl.py` (3-dim GL) | one-halo shear | `shear1h_explicit_gl/vals` | 149 ms/sample; $4.9\times10^{-5}$ vs the adaptive reference |
| `0d` | `python/0d/shear1h2h_max_explicit_gl.py` (3-dim GL) | 1h+2h max model | `shear1h2h_max_explicit_gl/vals` | z-RESOLVED weight (`explicit_mass_z_weights`) × `MaxMixtureProfile`; the explicit reference the max-model GL backends validate against |
| `0d` | `python/0d/validate_explicit_vs_production.py` | validator | — | explicit fixed-GL vs production |

Both read `miscentering/{f_mis,tau_mis}` strictly — no in-code default
fallback — and the max module requires `compute_lensing_2h = T`
(dSigma_hh NaNs sanitized to 0, exact under the max).

### GL nodes and weights

All fixed grids use standard Gauss–Legendre quadrature: nodes and
weights on $[-1,1]$ affine-mapped to the interval,
$x' = \tfrac{b-a}{2}x + \tfrac{b+a}{2}$, $w' = \tfrac{b-a}{2}w$
(`shared/datablock_models.py::gl_nodes`, via
`numpy.polynomial.legendre.leggauss`; the C++ twin
`y3_pipelines::gl_nodes` in the pipeline-owned
`shared/sel_gl_weights.hh` computes the same Legendre roots by Newton
iteration and applies the same map). Mass: `n_lnm` nodes (default 96) on
$[\ln M_{\rm low}, \ln M_{\rm high}]$; redshift: `n_z` nodes (default
64) on $[z_{\rm low}, z_{\rm high}]$. The composed shear weight is
$W(b;k,q) = w^z_q\,\tfrac{dV}{d\Omega dz}(z_q)\,\Omega(z_q)\,
\Sigma_{\rm crit}^{-1}(z_q)\, n(\ln M_k, z_q)\,S(b;\ln M_k,z_q)$; the
one-halo path sums it over $q$ once per sample, the max model keeps it
$z$-resolved.

The true-richness nodes are feature-placed: the shifted-Poisson
$P_{\rm HOD}$ is a near-delta ridge at low mass, so `n_q` GL nodes
(default 32) are laid **per $(\ln M, z)$ node** on the HOD support
$[\max(0,\mu_{\rm eff}-L_\lambda\sigma_{\rm eff}),\;
\mu_{\rm eff}+L_\lambda\sigma_{\rm eff}]$, with
$\mu_{\rm eff}=\lambda_{\rm cen}+\mu_{\rm sat}$,
$\sigma_{\rm eff}=\sqrt{\mu_{\rm sat}+(\sigma_\lambda\mu_{\rm sat})^2}$,
$L_\lambda$ = `l_lam` (default 6)
(`shared/sel_function.py::_compute_lam_nodes_and_P_HOD`). Placing the
panel on the integrand feature is why the fixed grids reach 4.9e-5
where a naive global quadrature would need adaptive subdivision.
Convergence is certified by doubled-node self-convergence
(96→192, 64→128, 32→64 nodes and $L_\lambda$ 6→8) in
`src/pipelines/des_y3/validate_against_fiducial.py`.

### Moment-expanded radial series

(Formerly `radial_series`; no runtime integration — offline tables plus
population moments.) Replaces the mass-dependent radial profile
evaluation with an offline expansion around the population mean of the
scale-radius coordinate. It is a candidate speed approximation for
one-halo shear, not a general profile emulator.

**Strong model limitation.** The committed table uses a fixed NFW
concentration, $c(M,z)=4$: no concentration–mass or
concentration–redshift evolution in the tabulated dimensionless
profile. The scale radius still varies with mass through
$r_s\propto M^{1/3}$, but a production $c(M,z)$ relation changes both
the amplitude and radial shape, and exact redshift contraction of the
population weights does not restore that missing dependence. Direct
radial evaluation is the correct choice when the production profile is
required.

Mathematical construction: let

$$
y=\ln r_s(M),\qquad x=R e^{-y},\qquad
x_{\rm mis}=\tau e^{-y},
$$

and write the fixed profile as
$\Delta\Sigma(R,y,\tau)=A_0(y)\,u(x,x_{\rm mis})$ with
$A_0(y)\propto e^y$. For a population with mean $\bar y$, define

$$
U_\ell(x,x_{\rm mis})=
\frac{1}{\ell!A_0(y)}
\left.\frac{\partial^\ell}{\partial s^\ell}
\left[A_0(y+s)\,u\!\left(Re^{-(y+s)},\tau e^{-(y+s)}\right)\right]
\right|_{s=0}.
$$

Because $A_0\propto e^y$, only the two dimensionless coordinates are
needed. With
$L=\partial/\partial\ln x + \partial/\partial\ln x_{\rm mis}$,
the stored functions are $U_0=u$, $U_1=(1-L)u$,
$U_2=\tfrac12(1-2L+L^2)u$, $U_3=\tfrac16(1-3L+3L^2-L^3)u$. The mass
integral is approximated with population moments,

$$
O(R)\approx N A_0(\bar y)\left[U_0+\mu_2U_2+\mu_3U_3\right],
$$

with $U_0$, $U_2$, $U_3$ interpolated at
$(\ln R-\bar y,\ln\tau-\bar y)$; $U_1$ is retained for validation (its
coefficient vanishes for the centered expansion used here). At runtime
the exact fixed-GL redshift weight supplies $N$, $\bar y$, $\mu_2$,
$\mu_3$.

| Dims | Backend | Cost | Precision vs 3d |
| --- | --- | ---: | --- |
| `0d` | Python (tables + moments) | 6--7 ms/sample | 56--86% (known fixed-c=4 defect; see link below); $3.7\times10^{-3}$ internal fixed-profile consistency (separate baseline) |
| `0d` | C++ (tables + moments) | 7 ms/sample | Same vs-3d defect; $3.7\times10^{-3}$ plus $1.6\times10^{-4}$ interpolation difference vs Python (separate baselines) |

These numbers measure internal consistency and the fixed-profile
approximation; they do not certify agreement with the production
profile. The known raw-$\Delta\Sigma$ mismatch is documented in
[`docs/known_issues/radial_series_vs_full_ltmz_defect.md`](../../../../docs/known_issues/radial_series_vs_full_ltmz_defect.md);
the table data are documented in
[`data/radial_series/README.md`](../../../../data/radial_series/README.md).

## The 3d backends

(Formerly `full_ltmz`; three adaptive dimensions.) The independent
references: true richness, true redshift, and halo mass stay explicit
and every selection kernel is evaluated at the quadrature nodes,

$$
O_{ij}(R)=\int dz\,d\ln M\,d\lambda_{\rm tr}\;
n\,\frac{dV}{d\Omega dz}\,\Omega\,
\Sigma_{\rm crit}^{-1}\,S_j S_i P_{\rm HOD}
\left[(1-f_{\rm mis})\Delta\Sigma_{\rm cen}
 +f_{\rm mis}\Delta\Sigma_{\rm mis}\right],
$$

one adaptive Cuhre (C++) or PAGANI (CUDA) triple integral per
(bin, R). The adaptive Python reference in `shared/explicit_grid_core.py`
shares the inner fixed-GL selection contraction and adaptively
subdivides the outer mass integral (reported error at or below 1e-6).

The max-model observable has its explicit adaptive reference here too
(`Shear1h2hMax3d`): because the two-halo term is $z$-dependent
and the max nonlinear, the whole integrand rides inside the adaptive
volume with $d_{\rm tot} = \max(\text{1h mixture},\,
b\cdot\Delta\Sigma_{hh})$. Both max backends require
`compute_lensing_2h = T` and sanitize `dSigma_hh` NaNs to 0 before
interpolation (exact under the max), and all backends here read
`miscentering/{f_mis,tau_mis}` strictly.

| Dims | Backend | Cost | Precision vs 3d |
| --- | --- | ---: | --- |
| `3d` | Python adaptive reference (`shared/explicit_grid_core.py`) | 35 s/sample | Reference (3d); reported error $\le 10^{-6}$ |
| `3d` | C++ Cuhre, `eps_rel=1e-4` | 51 s/sample | $3.3\times10^{-4}$ vs the 3d Python reference |
| `3d` | CUDA PAGANI, A100 | 32 s/sample | $3.4\times10^{-4}$ (baseline: the 3d C++ twin) |

These explicit implementations are reference tools; the maintained
production timing is supplied by the `0d` fixed-GL C++ backends.

## Running the backends

| Dims | Language | Sources | Module / output |
| --- | --- | --- | --- |
| `0d` | Python | `python/0d/shear1h_gl.py` (+ `validate_fast_vs_production.py`) | `shear1h_gl/vals` |
| `0d` | C++ | `cpp/0d/Shear1hGl.cc` (physics `cpp/0d/shear1h_gl_t.hh`) | `Shear1hGl.so`, section `shear1h_gl` |
| `0d` | Python | `python/0d/shear1h2h_max.py` (+ `validate_shear1h2h_max.py`) | `shear1h2h_max/vals` |
| `0d` | C++ | `cpp/0d/Shear1h2hMax.cc` (physics `cpp/0d/shear1h2h_max_t.hh`) | `Shear1h2hMax.so`, section `shear1h2h_max` |
| `0d` | CUDA | `cuda/0d/Shear1h2hMaxGpu.cu` (physics `cuda/0d/shear1h2h_max_gpu_t.cuh`) | `Shear1h2hMaxGpu.so` in `des_y3_shear1h_0d_cuda/`, section `shear1h2h_max_gpu` |
| `0d` | Python | `python/0d/shear1h_explicit_gl.py`, `python/0d/shear1h2h_max_explicit_gl.py` (+ `validate_explicit_vs_production.py`) | `shear1h_explicit_gl/vals`, `shear1h2h_max_explicit_gl/vals` |
| `0d` | Python/C++ | radial series: `python/0d/{generate_radial_series_tables,nfw_profile_family,shear1h_radial_series,validate_radial_series}.py`; `cpp/0d/Shear1hRadialSeries.cc` (physics `cpp/0d/shear1h_radial_series_t.hh`) | `shear1h_radial_series/vals` |
| `3d` | C++ | `cpp/3d/Shear1h3d.cc`, `cpp/3d/Shear1h2hMax3d.cc` (physics `cpp/3d/shear1h_3d_t.hh`, `cpp/3d/shear1h2h_max_3d_t.hh`) | `shear1h3d/…`, `shear1h2hmax3d/…` in `des_y3_shear1h_3d_cpp/` |
| `3d` | CUDA | `cuda/3d/Shear1h3dGpu.cu` (physics `cuda/3d/shear1h_3d_gpu_t.cuh`) | `shear1h3dgpu/…` in `des_y3_shear1h_3d_cuda/` |

The `0d` C++ modules build into one binary dir,
`release-build/src/modules/des_y3_shear1h_0d_cpp/` (three `.so`). No
CUDA backend exists for the 1h z-contracted path or the radial series
(a 1-D contraction / a few table lookups are not useful GPU targets).
Module labels are the ini `[section]` names (`[Shear1hGl]`,
`[Shear1h2hMax]`, `[Shear1hRadialSeries]`, ...), so pipelines drive
these backends by pointing those sections at the `.so` paths above.
