# Appendix: historical and retired code

Status map of everything under `src/modules/` that is **not** part of the
reference path, so no directory is unexplained. One line each; none of
this shapes the main documentation. (Inventory verified against
`src/modules/CMakeLists.txt`, 2026-08-10.)

## Registered, historical / survey-specific

Built by default but belonging to earlier analyses (DES Y1, SDSS,
mock/snapshot validation) or kept as GPU benchmarks:

| Module dir | Status / purpose |
|---|---|
| `gt_mock_cpu` / `gt_mock_gpu` | mock-catalogue $\gamma_t$ suites (avgCent/avgMisc variants) |
| `gt_card_cpu` / `gt_card_gpu` / `gt_card_triax_gpu` | Cardinal-simulation $\gamma_t$ suites; triaxial GPU variant |
| `gt_park_sel_cpu` / `gt_park_sel_gpu` | Park-style selection $\gamma_t$ suites |
| `sigma_park_y1` | GPU Σ (Y1); **broken** — awaits `lc_lt.cuh` debugging |
| `sigma_buzzard_y3` | GPU Σ on Buzzard (historical) |
| `sigma_mort_y1`, `mass_mort_y1`, `mass_y1` | Y1 Σ/mass integrands |
| `y1_analysis` | Y1 analysis integrands (incl. `_mor_2022` variants) |
| `sdss_analysis` | SDSS analysis integrands |
| `snapshotsim` | snapshot-simulation NC/Σ integrands |
| `buzzard_test` | `buzzard_sigma_halos` test module |
| `cluster_abundance_covariance` | abundance covariance |
| `ExampleScalar` / `ExampleVector` / `ExampleOneD` | macro-pattern examples |

## Registered but commented out

Present in `CMakeLists.txt` as comments ("historical"):
`DESxSPTModule`, `sigma_kappa_y1`, `compton_y_sims`, `model_sigmahm`.

## On disk, not registered

Eleven directories exist under `src/modules/` without a
`add_subdirectory` entry. Three of them are **Python-driven pipeline
steps** used by the reference path (loaded directly by file path from the
ini, so they need no CMake registration): `cp_camb`, `sel_function`,
`average_sigma_crit_inv`.

The rest are orphans/experiments:

| Directory | Note |
|---|---|
| `cuda` | CUDA experiments |
| `deltasigma` | superseded ΔΣ module |
| `finish` | pipeline finisher experiment |
| `mass_conversion` | mass-definition conversion |
| `n_operator_ratios` | $N[M]/N[1]$, $N[b]/N[1]$ finisher (optional diagnostics; not in the reference module list) |
| `parabola` | toy/example |
| `prj_lens_model` | earlier projection-lensing model |
| `red_shear_prj` | legacy name/location of the projection branch — the physics now lives in `sigma_prj_cpu` under the `shear_prj` section |

## Known-broken legacy paths

- **`SigmaTotSel` / `DSigmaTotSel`** (the wired fiducial 1h+2h
  composition): verified broken 2026-08-10. Two independent defects:
  (i) with `compute_lensing_2h = T`, `halo_model` publishes
  `haloModel/dSigma_hh` with NaN below $R \approx 8.6\,h^{-1}$cMpc
  (the cluster_toolkit Hankel ΔΣ output on the `Rp` grid); (ii) the
  `SigmaTotWeight`/`DSigmaTotWeight` structs
  (`src/modules/num_counts_sel/lensing_weights.hh`) build their 2h
  `Interp2D` on the `haloModel/r_sigma` axis (0.1–20) while the table
  is computed on the `Rp` grid (1–35) — a silent axis mismatch, so even
  the finite values are evaluated at the wrong radius. Any 1h+2h result
  must currently be assembled from `BiasWeightedSel` + `xi_nl` directly
  (as the comparison figure in the science chapter does).

## Retired types and names

- **`HMB_t`** (C++ halo-bias type) and the `tinker_bias` module —
  retired; $b(M, z)$ is read from the DataBlock section
  `haloModel/bias`, published by `halo_model_cosmosis.py`.
- **`prj_params` as a standalone module** — retired; `bsel.py` imports
  the EMG coefficients directly from `y3_buzzard/prj_params.py`.
- **`red_shear_prj`** as a module name — legacy; standardize on
  `shear_prj`.
- **Reduced shear $g_t$** — retired observable (see
  {doc}`../math/index`).
