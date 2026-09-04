# Running the reference pipeline

The reference configuration of the DES Y3 cluster-cosmology analysis is
[`cosmosis-models/des_y3.ini`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/cosmosis-models/des_y3.ini)
in this repository. It keeps the forward model and DataBlock contract of
the DES Y1 pipeline ({doc}`variants`) and runs the three observable
stages through their `src/pipelines/des_y3` fixed Gauss–Legendre (`0d`)
implementations — `NumCountsSijGl`, `Shear1hGl`, `ShearPrjGl` — the
"Recommended methods" of
[`src/pipelines/des_y3/README.md`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/pipelines/des_y3/README.md).
Cosmology, halo model, selection function, and the selection-bias
operators are unchanged: this is an implementation swap, not a different
theory vector.

The software suite this pipeline is built from — the CosmoSIS module
pattern, the model/integrand separation, and the number-count and
population-averaged lensing forward model — is described in
[DES Cluster et al. 2023](https://ui.adsabs.harvard.edu/abs/2023arXiv230906593A/abstract)
(arXiv:[2309.06593](https://arxiv.org/abs/2309.06593)), the main
reference for this documentation. The optical selection-bias and
projection-lensing extension is
[Costanzi et al. 2026](https://ui.adsabs.harvard.edu/abs/2026PhRvD.113j3508C/abstract)
(arXiv:[2604.05833](https://arxiv.org/abs/2604.05833)).

## The pipeline

```ini
[pipeline]
modules = consistency GrowthFactor cp_camb MfTinker halo_model
          average_sigma_crit_inv sel_function
          NumCountsSijGl Shear1hGl
          b_sel_marg bsel
          ShearPrjGl
          likelihoods
values = ${DES_CLUSTER_NERSC_DIR}/cosmosis-models/mock_mcmc_widePlanck_values.ini
likelihoods = likelihoods
timing = T
```

| # | Module | What it computes | Language · source | Cost/sample |
|---|---|---|---|---:|
| 1 | {doc}`consistency <cosmology/consistency>` | completes the cosmological parameter set | Python · CosmoSIS Standard Library | <1 ms |
| 2 | {doc}`GrowthFactor <cosmology/growth_factor>` | linear growth $D(z)$, $f(z)$ | C · CosmoSIS Standard Library | <1 ms |
| 3 | {doc}`cp_camb <cosmology/cp_camb>` | linear $P(k,z)$ (CosmoPower emulator) + distances | Python · `src/modules/cp_camb` | 4 ms |
| 4 | {doc}`MfTinker <cosmology/mf_tinker>` | Tinker halo mass function | Fortran · CosmoSIS Standard Library | 155 ms |
| 5 | {doc}`halo_model <cosmology/halo_model>` | Tinker bias $b(M,z)$, $\xi_{\rm NL}$, NFW lensing tables | Python · `y3_buzzard/halo_model_cosmosis.py` (physics in `src/pipelines/cosmology`) | 141 ms |
| 6 | {doc}`average_sigma_crit_inv <cosmology/sigma_crit_inv>` | $\langle\Sigma_{\rm crit}^{-1}\rangle(z_l)$ | Python · `src/modules/average_sigma_crit_inv` | <1 ms |
| 7 | {doc}`sel_function <systematics/sel_function>` | selection tensor $S_{ij}(\ln M, z)$ | Python · `src/pipelines/systematics/selection_richness` | 197 ms |
| 8 | {doc}`NumCountsSijGl <observables/number_counts>` | cluster counts $N_i[1]$ (`0d`) | C++ · `src/pipelines/des_y3/number_counts` | 6 ms |
| 9 | {doc}`Shear1hGl <observables/shear_halo>` | one-halo shear with miscentering (`0d`) | C++ · `src/pipelines/des_y3/shear_1h2h` | 9 ms |
| 10 | {doc}`b_sel_marg <systematics/bsel>` | selection-bias operators $(P_1, I_1, J)$ | C++ · `src/modules/b_sel_marg_cpu` | 66 ms |
| 11 | {doc}`bsel <systematics/bsel>` | bias plateaus $(B_{\rm small}, B_{\rm large})$ | Python · `src/pipelines/systematics/selection_bias` | few ms (not pinned) |
| 12 | {doc}`ShearPrjGl <observables/shear_projection>` | projection shear $\gamma_t^{\rm prj}(R)$ (`0d`) | C++ · `src/pipelines/des_y3/shear_projection` | 154 ms |
| 13 | {doc}`likelihoods <observables/likelihood>` | Gaussian $\log L$ | Python · `src/pipelines/buzzard/likelihoods` | <1 ms |

Costs are the pinned Perlmutter CPU fiducial measurements from
`src/pipelines/des_y3/README.md` (`timing = T` in the ini prints them
per module on every run; on an Apple-Silicon build they come out 3–7×
faster). The whole pipeline is well under one second per sample; the
Python support stages (4, 5, 7) dominate, not the C++ observables.

```{note}
Stages 1–7 and 10–11 keep the same physics and DataBlock contract as the
DES Y1 pipeline ({doc}`variants`). Only 8, 9, and 12 differ, and each is
algorithmically identical to its DES Y1 counterpart — `NumCountsSijGl`
"by identity" with `NumCountsSel`, `Shear1hGl` bitwise-equal to
`Shear1hMisSel`, `ShearPrjGl` sharing the same `ShearPrjCore` as
`shear_prj_frozen_physics` but exact in $z$ — under the
`src/pipelines/des_y3` namespace, with their own output sections
(`numcounts_sij_gl`, `shear1h_gl`, `shear_prj_gl`) so both generations
can co-run in one pipeline for comparison.
```

Data flow (edge labels are the DataBlock sections passed between
modules; blue = cosmology quantities, orange = systematics,
green = cluster observables, grey = likelihood):

```{image} _static/img/pipeline_dataflow.png
:alt: Data flow of the des_y3.ini reference pipeline
:width: 100%
```

(Source: `docs/figs/pipeline_dataflow.mmd`; regenerate the PNG with
`npx -y @mermaid-js/mermaid-cli -i docs/figs/pipeline_dataflow.mmd -o
docs/source/_static/img/pipeline_dataflow.png -b white -s 2`.)

### Optional stages

Not in the reference module list, documented on their own pages:

- `costanzi_bprj` — publishes the bin grid for the
  $\mathcal B_{\rm prj}(R)$ selection-bias correction of the max model
  ({doc}`systematics/costanzi_bprj`); enable with
  `is_b_proj_costanzi26 = T` and `shear_max_section = shear1h2h_max` in
  `[likelihoods]`.
- `Shear1h2hMax` + `halo_model` with `compute_lensing_2h = T` — the
  traditional $\max(1h, b\,2h)$ shear model ({doc}`observables/second_halo_term`).
- `boost_factor` — publishes the McClintock et al. (2019) source-dilution
  boost $B(R)$ per bin ({doc}`systematics/boost_factor`).
- `prj_params` — publishes the frozen EMG projection-kernel coefficients
  into `plob_ltr_params`; only the adaptive `3d` observables need it
  ({doc}`variants`).

## Required repositories and environment

| Repository | Role | Env variable |
|---|---|---|
| [y3_cluster_cpp](https://github.com/estevesjh/y3_cluster_cpp) | C++/CUDA modules ({doc}`installation`) + Python modules; `src/pipelines/{des_y3,systematics,cosmology,shared,buzzard}` | `Y3_CLUSTER_CPP_DIR` |
| [des-nersc-cluster-scripts](https://github.com/estevesjh/des-nersc-cluster-scripts) | values file, data vectors, sbatch scripts (deployed on Perlmutter as `des-cluster-nersc`) | `DES_CLUSTER_NERSC_DIR` |
| [cosmosis-standard-library](https://github.com/joezuntz/cosmosis-standard-library) | `consistency`, `GrowthFactor`, `MfTinker` | `COSMOSIS_STANDARD_LIBRARY` |
| [camb-emulator](https://github.com/estevesjh/camb-emulator) | trained CosmoPower emulators read by `cp_camb` | (paths in the `[cp_camb]` section) |

On Perlmutter, `fast-cpu/setup_env.sh` in `des-nersc-cluster-scripts`
(sourced by every job script) performs the module swaps and exports; the
CosmoSIS environment comes from `setup-cosmosis-nersc` with the
`y3cl_je` conda env:

```bash
module swap cudatoolkit/12.9 cudatoolkit/12.2
module swap gcc-native/13.2 gcc-native/12.3
export Y3_CLUSTER_CPP_DIR=/pscratch/sd/j/jesteves/y3_cluster_cpp
export DES_CLUSTER_NERSC_DIR=/pscratch/sd/j/jesteves/github/des-cluster-nersc
export COSMOSIS_STANDARD_LIBRARY=/global/common/software/des/jesteves/cosmosis-standard-library
export PYTHONPATH=${Y3_CLUSTER_CPP_DIR}:${PYTHONPATH:-}
export OMP_NUM_THREADS=1
source ${COSMOSIS_REPO_DIR}/setup-cosmosis-nersc \
       /global/common/software/des/common/Conda_Envs/y3cl_je
```

The same pipeline runs CPU-only on macOS; see {doc}`building_macos`
for the verified recipe (mamba env, CosmoSIS/CUBA built from source,
`CMakePresets.json` `macos-cpu` preset). `Y3_CLUSTER_CPP_DIR` must point
at the checkout root in both cases — the modules use it to find `data/`
and the sibling `.so` files.

The three C++ observable modules build alongside every other module (no
separate step) via the {doc}`installation` recipe, into
`release-build/src/modules/des_y3_{numcounts,shear1h,shear_prj}_0d_cpp/`,
which is where the ini's `file =` lines point.

## Values, priors, data vector, covariance

Same as the DES Y1 pipeline ({doc}`variants`): `[pipeline] values`
points at
[`mock_mcmc_widePlanck_values.ini`](https://github.com/estevesjh/des-nersc-cluster-scripts/blob/main/cosmosis-models/mock_mcmc_widePlanck_values.ini)
(10 varied parameters, 5 cosmology + 5 HOD, flat priors from the
`[min start max]` boxes, no separate priors file), and `[likelihoods]
filename` at the Buzzard mock data vector `mock_dv_buzzard.npz`. The
`[miscentering]` section of the values file must publish `f_mis` and
`tau_mis` — the des_y3 modules have no in-code fallback. The
repository's own
`cosmosis-models/mock_mcmc_widePlanck_values_mis.ini` is a
self-contained copy (widePlanck + the Y3 fiducial $f_{\rm mis}=0.22$,
$\tau_{\rm mis}=0.17$) used by the benchmark inis below.

## Commands

Smoke test — single sample, `test` sampler, per-module timings printed:

```bash
cd ${Y3_CLUSTER_CPP_DIR}
cosmosis cosmosis-models/des_y3.ini
```

The DataBlock dump lands in `cosmosis-models/des_y3_test_output/`
(gitignored). Production sampling overrides the sampler on the command
line:

```bash
cosmosis cosmosis-models/des_y3.ini -p runtime.sampler=polychord runtime.resume=F polychord.live_points=500
```

Other inis in `cosmosis-models/` (all read the same values file and are
gitignored-output like `des_y3.ini`):

- `des_y3_cpp_python.ini`, `des_y3_cpp_gpu.ini` — run *every* backend of
  both strategies (fixed-GL `0d`; adaptive `3d`) for cpp+python and
  cpp+gpu respectively, all in one pipeline so the families share one
  cosmology/halo-model sample and can be timed and cross-checked. The
  `3d` legs reduce the shear walls to a few corner points (minutes, not
  hours, per sample); `des_y3_cpp_gpu.ini` skips GPU number counts
  entirely (no 0d CUDA number-counts module exists — see its header).
  Results: "Precision and cost overview" in
  `src/pipelines/des_y3/README.md`.
- `*_apriori.ini` — the same, over hundreds of prior draws, recording
  cost and success per draw (about 30% of draws are cheap, deliberate
  `cp_camb` rejections outside the emulator's trained box; adaptive
  `3d` cost varies 40–60× across the prior).
- `real_pipeline_extract*.ini` — fixture dumps at the fiducial point for
  the offline validators and tests.

## What the likelihood compares

$$\log L = -\tfrac12\Big[
\delta_{\rm NC}^{\mathsf T} C_{\rm NC}^{-1} \delta_{\rm NC}
+ \delta_{\gamma}^{\mathsf T} C_{\gamma}^{-1} \delta_{\gamma}\Big],
\qquad
\gamma_t^{\rm theory}(R \mid i) =
\frac{N_i[\gamma_t^{1h,\rm full}](R)}{N_i[1]} + \gamma_t^{\rm prj,\,cl}(R \mid i),$$

12 number counts + 180 shear points, unchanged from the DES Y1 pipeline
— identical theory-vector definition, only the modules computing
$N_i[1]$, $N_i[\gamma_t^{1h}]$, and $\gamma_t^{\rm prj}$ differ. The
projection term entering the likelihood is the **clustered channel**
`cl` (the correlated excess $\Sigma^{\rm prj}$; the `rnd` background
channel is a diagnostic — {doc}`math/index`). Details:
{doc}`observables/likelihood`.

`src/pipelines/buzzard/likelihoods/likelihood_cp.py` takes the DataBlock
section names of the three observables as options; `des_y3.ini` sets
them to `numcounts_sij_gl`, `shear1h_gl`, `shear_prj_gl`, so the
des_y3 modules coexist with the DES Y1 ones without DataBlock
overwrites. The optional `shear_max_section` switches the shear theory
to the traditional max model (`shear1h2h_max`, no projection term) and
`is_b_proj_costanzi26 = T` applies the Costanzi-2026
$\mathcal{B}_{\rm prj}(R)$ correction to it
({doc}`systematics/costanzi_bprj`, {doc}`variants`).
