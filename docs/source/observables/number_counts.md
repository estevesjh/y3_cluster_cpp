# Cluster Number Counts

`C++` · `y3_cluster_cpp` (`src/pipelines/des_y3`) · `Cluster observable` · module `NumCountsSijGl` · `~6 ms/sample`

Computes the expected cluster count $N_i[1]$ in each of the 12
$(\lambda^{\rm ob}, z^{\rm ob})$ bins by integrating the halo mass function
against the pre-tabulated selection function. Its output is the first block
of the theory vector and the denominator of the stacked one-halo shear.

## Script

- Model: [`src/pipelines/shared/sel_gl_weights.hh`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/pipelines/shared/sel_gl_weights.hh)
  (`y3_pipelines::SelGlWeights` — the pipeline-owned fixed
  Gauss-Legendre mass-weight builder shared with {doc}`shear_halo`,
  identity-certified against the production engine `SelGLCore` in
  `src/models/n_operator_sel_gl_t.hh`, which `NumCountsSel.so` itself
  still uses).
- Module driver: [`src/pipelines/des_y3/number_counts/cpp/0d/NumCountsSijGl.cc`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/pipelines/des_y3/number_counts/cpp/0d/NumCountsSijGl.cc)
  (`DEFINE_COSMOSIS_SCALAR_EVALUATOR_MODULE`) — the des_y3-namespaced
  wrapper, algorithmically identical to `NumCountsSelGL` ("by identity",
  {doc}`../variants`), with its own module label and output section so
  the two can co-run in one pipeline for comparison.
- Compiled library loaded by CosmoSIS:
  `${Y3_CLUSTER_CPP_DIR}/release-build/src/modules/des_y3_numcounts_0d_cpp/NumCountsSijGl.so`.

## Numerical framework

The full integral — this module is the $f = 1$ instance of the
population operator

$$N_i[f] = \int d\ln M \int dz\;
\Omega(z)\,\frac{dV}{d\Omega\,dz}\,\frac{dn}{d\ln M}(M,z)\,
S_{ij}(\ln M, z)\, f(\ln M, z),$$

the number-count forward model of
[DES Cluster et al. 2023](https://ui.adsabs.harvard.edu/abs/2023arXiv230906593A/abstract)
(arXiv:[2309.06593](https://arxiv.org/abs/2309.06593), its Eq. 1) — the
reference paper for this software suite —
with $\Omega(z)$ the survey area ({doc}`../modules/survey_area`) and
$S_{ij}$ the tabulated selection function
({doc}`../systematics/sel_function`). Evaluation is fixed Gauss–Legendre:
the $z$ axis is contracted once per sample into per-bin mass weights,
and each count is one 1-D mass sum — a deterministic 6 ms per sample vs
the retired adaptive Cuhre path (mean 0.11 s, tail 1 s),
grid-convergence error $< 0.05\%$. **The complete step-by-step recipe
lives in {doc}`../numerics/index`, §"The number-counts and one-halo
lensing recipe, step by step".** Derivation: {doc}`../math/index`.

## CosmoSIS setup

```ini
[NumCountsSijGl]
file = ${Y3_CLUSTER_CPP_DIR}/release-build/src/modules/des_y3_numcounts_0d_cpp/NumCountsSijGl.so
bin_index = 0 1 2 3 4 5 6 7 8 9 10 11
zt_low  = 0.05
zt_high = 0.80
lnm_low  = 29.9336
lnm_high = 36.7300
n_lnm = 96
n_z   = 64
```

- Build once (see {doc}`../installation`); requires `Y3_CLUSTER_CPP_DIR`
  so the `.so` and its data files resolve.
- Ordering: after `sel_function`, `MfTinker`, `cp_camb`.
- No adaptive-Cuhre knobs (`algorithm`/`eps_*`/`max_eval`/
  `use_cartesian_product`) — this evaluator is fixed Gauss-Legendre only;
  `n_lnm`/`n_z` are its node counts.

## Configuration options

| Option | Meaning | Units | Reference value |
|---|---|---|---|
| `bin_index` | wall of bins to evaluate (richness index fastest) | — | `0 … 11` |
| `zt_low`, `zt_high` | true-redshift integration limits | — | 0.05, 0.80 |
| `lnm_low`, `lnm_high` | mass integration limits | $\ln(M_\odot/h)$ | 29.9336, 36.7300 |
| `n_lnm`, `n_z` | GL nodes in $\ln M$ / $z$ | — | 96, 64 (defaults) |

## DataBlock inputs

| DataBlock input | Meaning | Units / shape | Produced by |
|---|---|---|---|
| `sel_function/{lnM, z, S_stack}` | selection tensor $S_{ij}(\ln M, z)$ | `(12, 64, 192)` | `sel_function` |
| `mass_function/{m_h, z, dndlnmh}` | halo mass function (queried through `HMF_t`, which applies the $\Omega_m - \Omega_\nu$ mass-axis shift) | $h^3\,\mathrm{Mpc}^{-3}$ | `MfTinker` |
| `cluster_abundance/{hmf_s, hmf_q}` | HMF nuisance amplitudes | scalars | sampler (values file) |
| `distances/{z, d_a}` | comoving volume element via `DV_DO_DZ_t` | Mpc | `cp_camb` |
| `cosmological_parameters/{omega_m, omega_nu}` | HMF mass-axis shift | scalars | `consistency` |

## DataBlock outputs

| DataBlock output | Meaning | Units / shape | Consumed by |
|---|---|---|---|
| `numcounts_sij_gl/vals` | expected counts $N_i[1]$ per bin | `(12,)` | `likelihoods` (data block and shear normalisation) |

The output section name is hard-coded in the module (deliberately not an
ini knob: a CosmoSIS `[DEFAULT]` block would propagate an
`output_section` value into every module and silently redirect writes).
DES Y1's `NumCountsSel.so` writes `numcountssel/vals` instead — the two
sections never collide, so both can run in the same pipeline for
comparison ({doc}`../variants`).

## DES Y3 implementations

This module is the `0d` fast cell (zero adaptive dimensions; formerly
`fast_mass`) — the reference pipeline's choice per the "Recommended
methods" table of `src/pipelines/des_y3/README.md`. The other
implementations under `src/pipelines/des_y3/number_counts` follow the
layout of {doc}`../pipeline_organization`; costs are per sample on the
12-bin wall (Perlmutter CPU, A100 for CUDA), precision is quoted
against the adaptive `3d` Python reference (reported integration error
$\le 10^{-6}$):

| Dims | Backend | Implementation and status | Cost | Precision vs 3d |
|---|---|---|---:|---|
| `3d` | Python | `numcounts3d` — adaptive explicit $(\lambda_{\rm true},\ln M,z)$ reference (needs the `prj_params` stage for `plob_ltr_params` — {doc}`../variants`) | 25 s | reference |
| `3d` | C++ | `NumCounts3d.so`; adaptive Cuhre | 3.1 s | 1.1e-4 |
| `3d` | CUDA | `NumCounts3dGpu.so`; PAGANI on A100 | 2.0 s | 1.2e-4 (2.1e-5 vs the C++ twin) |
| `0d` | Python | `numcounts_explicit_gl.py` — explicit fixed-GL 3-dim grid | 83 ms | 3.5e-5 |
| `0d` | Python | `numcounts_sij_gl.py` — $S_{ij}$-tabulated 2-dim GL fast sum | 5 ms | 7.6e-4 (2.4e-15 vs production) |
| `0d` | **C++ (this page)** | `NumCountsSijGl.so` — same fast sum, algorithmically identical to DES Y1's `NumCountsSel.so` | 6 ms | 7.6e-4 (identity with production) |

A radial-series reduction does not apply because the counts operator
has no radial profile ($f=1$). The $7.6\times10^{-4}$ of the fast path
is the cost of tabulating $S_{ij}$ on the shared `sel_function` grid
rather than evaluating the richness kernels at every node; it is the
dominant approximation of the production counts. Agreement with
`NumCountsSel.so` is recorded separately as an identity check.
Validators: `validate_fast_vs_production.py`,
`validate_explicit_vs_production.py` next to the Python backends;
tests in {doc}`../testing/src_pipelines_des_y3`.
