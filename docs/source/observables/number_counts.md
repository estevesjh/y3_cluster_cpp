# Cluster Number Counts

Computes the expected cluster count $N_i[1]$ in each of the 12
$(\lambda^{\rm ob}, z^{\rm ob})$ bins by integrating the halo mass function
against the pre-tabulated selection function. Its output is the first block
of the theory vector and the denominator of the stacked one-halo shear.

This page documents the `fast_mass` C++ implementation (`NumCountsFastMass`)
specifically — see "DES Y3 implementations" below for the other backends.

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
({doc}`../selection/sel_function`). Evaluation is fixed Gauss–Legendre:
the $z$ axis is contracted once per sample into per-bin mass weights,
and each count is one 1-D mass sum — deterministic 0.02 s per sample vs
the retired adaptive Cuhre path (mean 0.11 s, tail 1 s),
grid-convergence error $< 0.05\%$. **The complete step-by-step recipe
lives in {doc}`../numerics/index`, §"The number-counts and one-halo
lensing recipe, step by step".** Derivation: {doc}`../math/index`.

## DES Y3 implementations

This module is the `fast_mass` cell — the reference pipeline's choice
per `src/pipelines/des_y3/README.md`'s own "Reference pipeline choices"
table. Other implementations follow the organization in
{doc}`../pipeline_organization`:

| Strategy | Backend | Implementation and status |
|---|---|---|
| `full_ltmz` | Python | Explicit $(\lambda_{\rm true},\ln M,z)$ reference; fixed GL |
| `full_ltmz` | C++ | `NumCountsFullLtmz.so`; adaptive Cuhre reference (needs an added `plob_ltr_params` stage — {doc}`../variants`) |
| `full_ltmz` | CUDA | `NumCountsFullLtmzGpu.so`; PAGANI reference |
| `fast_mass` | Python | Importable re-expression of the redshift contraction |
| `fast_mass` | **C++ (this page)** | `NumCountsFastMass.so` — algorithmically identical to DES Y1's `NumCountsSel.so` |

The implementations live below
`src/pipelines/des_y3/observables/number_counts`. `radial_series` does not
apply because the counts operator has no radial profile ($f=1$). Accuracy is
measured against `full_ltmz`; agreement with `NumCountsSel.so` is recorded
separately as an identity check — this module is expected to be bitwise
equal to it.

## Script

- Model: [`src/models/n_operator_sel_gl_t.hh`](https://github.com/estevesjh/y3_cluster_cpp/blob/pipelines/des_y3/src/models/n_operator_sel_gl_t.hh)
  (`nosel_gl_detail::SelGLCore` — the fixed Gauss-Legendre mass-weight
  builder shared with {doc}`shear_halo`).
- Module driver: [`src/pipelines/des_y3/observables/number_counts/fast_mass/cpp/NumCountsFastMass.cc`](https://github.com/estevesjh/y3_cluster_cpp/blob/pipelines/des_y3/src/pipelines/des_y3/observables/number_counts/fast_mass/cpp/NumCountsFastMass.cc)
  (`DEFINE_COSMOSIS_SCALAR_EVALUATOR_MODULE`) — the des_y3-namespaced
  wrapper, algorithmically identical to `NumCountsSelGL` ("by identity",
  {doc}`../variants`), with its own module label and output section so
  the two can co-run in one pipeline for comparison.
- Compiled library loaded by CosmoSIS:
  `${Y3_CLUSTER_CPP_DIR}/release-build/src/modules/des_y3_numcounts_fast_mass_cpp/NumCountsFastMass.so`.

## CosmoSIS setup

```ini
[NumCountsFastMass]
file = ${Y3_CLUSTER_CPP_DIR}/release-build/src/modules/des_y3_numcounts_fast_mass_cpp/NumCountsFastMass.so
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
| `numcounts_fast_mass/vals` | expected counts $N_i[1]$ per bin | `(12,)` | `likelihoods` (data block and shear normalisation) |

The output section name is hard-coded in the module (deliberately not an
ini knob: a CosmoSIS `[DEFAULT]` block would propagate an
`output_section` value into every module and silently redirect writes).
DES Y1's `NumCountsSel.so` writes `numcountssel/vals` instead — the two
sections never collide, so both can run in the same pipeline for
comparison ({doc}`../variants`).
