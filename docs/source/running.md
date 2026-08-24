# Running the reference pipeline

The reference configuration of the DES Y3 cluster-cosmology analysis is
[`cosmosis-models/des_y3.ini`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/cosmosis-models/des_y3.ini)
in this repository: the same forward model and DataBlock contract as
the DES Y1 pipeline ({doc}`variants`), with the three observable stages
swapped for their `src/pipelines/des_y3` fast_mass implementations —
`NumCountsFastMass`, `Shear1hFastMass`, `ShearPrjFastMass` — per
[`src/pipelines/des_y3/README.md`](https://github.com/estevesjh/y3_cluster_cpp/blob/pipelines/des_y3/src/pipelines/des_y3/README.md)'s
own "Reference pipeline choices" table. Cosmology, halo model,
selection function, and the selection-bias operators are unchanged:
this is an implementation swap, not a different theory vector.

The software suite this pipeline is built from — the CosmoSIS module
pattern, the model/integrand separation, and the number-count and
population-averaged lensing forward model — is described in
[DES Cluster et al. 2023](https://ui.adsabs.harvard.edu/abs/2023arXiv230906593A/abstract)
(arXiv:[2309.06593](https://arxiv.org/abs/2309.06593)), the main
reference for this documentation.

## The pipeline

```ini
[pipeline]
modules = consistency GrowthFactor cp_camb MfTinker halo_model
          average_sigma_crit_inv sel_function
          NumCountsFastMass Shear1hFastMass
          b_sel_marg bsel
          ShearPrjFastMass
          likelihoods
values = ${DES_CLUSTER_NERSC_DIR}/cosmosis-models/mock_mcmc_widePlanck_values.ini
likelihoods = likelihoods
```

| # | Module | What it computes | Language · source |
|---|---|---|---|
| 1 | {doc}`consistency <cosmology/consistency>` | completes the cosmological parameter set | Python · CosmoSIS Standard Library |
| 2 | {doc}`GrowthFactor <cosmology/growth_factor>` | linear growth $D(z)$, $f(z)$ | C · CosmoSIS Standard Library |
| 3 | {doc}`cp_camb <cosmology/cp_camb>` | linear $P(k,z)$ (CosmoPower emulator) + distances | Python · `y3_cluster_cpp` |
| 4 | {doc}`MfTinker <cosmology/mf_tinker>` | Tinker halo mass function | Fortran · CosmoSIS Standard Library |
| 5 | {doc}`halo_model <cosmology/halo_model>` | Tinker bias $b(M,z)$, $\xi_{\rm NL}$, NFW lensing tables | Python · `y3_cluster_cpp` |
| 6 | {doc}`average_sigma_crit_inv <cosmology/sigma_crit_inv>` | $\langle\Sigma_{\rm crit}^{-1}\rangle(z_l)$ | Python · `y3_cluster_cpp` |
| 7 | {doc}`sel_function <selection/sel_function>` | selection tensor $S_{ij}(\ln M, z)$ | Python · `systematics/selection_richness` |
| 8 | {doc}`NumCountsFastMass <observables/number_counts>` | cluster counts $N_i[1]$ (fast_mass, des_y3) | C++ · `y3_cluster_cpp` |
| 9 | {doc}`Shear1hFastMass <observables/shear_halo>` | one-halo shear with miscentering (fast_mass, des_y3) | C++ · `y3_cluster_cpp` |
| 10 | {doc}`b_sel_marg <selection/bsel>` | selection-bias operators $(P_1, I_1, J)$ | C++ · `y3_cluster_cpp` |
| 11 | {doc}`bsel <selection/bsel>` | bias plateaus $(B_{\rm small}, B_{\rm large})$ | Python · `systematics/selection_bias` |
| 12 | {doc}`ShearPrjFastMass <observables/shear_projection>` | projection shear $\gamma_t^{\rm prj}(R)$ (fast_mass, des_y3) | C++ · `y3_cluster_cpp` |
| 13 | {doc}`likelihoods <observables/likelihood>` | Gaussian $\log L$ | Python · `y3_cluster_cpp` |

```{note}
Stages 1–7 and 10–11 keep the same physics and DataBlock contract as the DES
Y1 pipeline ({doc}`variants`). Their maintained DES Y3 Python entry points now
live under `src/pipelines/systematics/`; the legacy copies remain available.
Only 8, 9, and 12 differ, and each is
algorithmically identical to its DES Y1 counterpart — `NumCountsFastMass`
"by identity" with `NumCountsSel`, `Shear1hFastMass` bitwise-equal to
`Shear1hMisSel`, `ShearPrjFastMass` sharing the same `ShearPrjCore` as
`shear_prj_frozen_physics` — just under the `src/pipelines/des_y3`
namespace so both generations can co-run in one pipeline for comparison
(their output DataBlock sections never collide).
```

Data flow (edge labels are the DataBlock sections passed between
modules; blue = cosmology quantities, orange = selection effects,
green = cluster observables, grey = likelihood):

```{image} _static/img/pipeline_dataflow.png
:alt: Data flow of the des_y3.ini reference pipeline
:width: 100%
```

(Source: `docs/figs/pipeline_dataflow.mmd`; regenerate the PNG with
`npx -y @mermaid-js/mermaid-cli -i docs/figs/pipeline_dataflow.mmd -o
docs/source/_static/img/pipeline_dataflow.png -b white -s 2`.)

## Required repositories and environment

| Repository | Role | Env variable |
|---|---|---|
| [y3_cluster_cpp](https://github.com/estevesjh/y3_cluster_cpp) | C++/CUDA modules ({doc}`installation`) + Python modules, `src/pipelines/des_y3` (this pipeline's 3 swapped modules) | `Y3_CLUSTER_CPP_DIR=/pscratch/sd/j/jesteves/y3_cluster_cpp` |
| [des-nersc-cluster-scripts](https://github.com/estevesjh/des-nersc-cluster-scripts) | values file, data vectors, sbatch scripts (deployed as `des-cluster-nersc`) | `DES_CLUSTER_NERSC_DIR=/pscratch/sd/j/jesteves/github/des-cluster-nersc` |
| [cosmosis-standard-library](https://github.com/joezuntz/cosmosis-standard-library) | `consistency`, `GrowthFactor`, `MfTinker` | `COSMOSIS_STANDARD_LIBRARY` |
| [camb-emulator](https://github.com/estevesjh/camb-emulator) | trained CosmoPower emulators read by `cp_camb` | (paths in the `[cp_camb]` section) |

`fast-cpu/setup_env.sh` (sourced by every job script) performs the
Perlmutter module swaps and exports; the CosmoSIS environment comes from
`setup-cosmosis-nersc` with the `y3cl_je` conda env:

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

`NumCountsFastMass.so`, `Shear1hFastMass.so`, and `ShearPrjFastMass.so`
build alongside every other module — no separate build step — via the
normal {doc}`installation` recipe; they register in
`src/modules/CMakeLists.txt` under the `des_y3` block.

## Values, priors, data vector, covariance

Same as the DES Y1 pipeline ({doc}`variants`) — this ini's `[pipeline]
values` line points at the identical
[`mock_mcmc_widePlanck_values.ini`](https://github.com/estevesjh/des-nersc-cluster-scripts/blob/9fd24ddc075d394af4e20241bda716ac4d529fcb/cosmosis-models/mock_mcmc_widePlanck_values.ini):
10 varied parameters (5 cosmology + 5 HOD), flat priors from the
`[min start max]` boxes, no separate priors file. See {doc}`variants`
for the full parameter table, data-vector, and covariance details —
they don't change with this implementation swap.

## Commands

Smoke test (single sample, `test` sampler):

```bash
cd ${Y3_CLUSTER_CPP_DIR}
cosmosis cosmosis-models/des_y3.ini
```

Production sampling follows the same override pattern as the DES Y1
pipeline ({doc}`variants`) — e.g.
`-p runtime.sampler=polychord runtime.resume=F polychord.live_points=500 ...` —
substituting `des_y3.ini` for `mock_mcmc_buzzard.ini`.

## What the likelihood compares

$$\log L = -\tfrac12\Big[
\delta_{\rm NC}^{\mathsf T} C_{\rm NC}^{-1} \delta_{\rm NC}
+ \delta_{\gamma}^{\mathsf T} C_{\gamma}^{-1} \delta_{\gamma}\Big],
\qquad
\gamma_t^{\rm theory}(R \mid i) =
\frac{N_i[\gamma_t^{1h,\rm full}](R)}{N_i[1]} + \gamma_t^{\rm prj}(R \mid i),$$

12 number counts + 180 shear points, unchanged from the DES Y1 pipeline
— identical theory-vector definition, only the modules computing
$N_i[1]$, $N_i[\gamma_t^{1h}]$, and $\gamma_t^{\rm prj}$ differ. Details:
{doc}`observables/likelihood`. The traditional $1h{+}2h$ composition
(`Shear1h2hMax`) is available as a model option, not part of this
reference pipeline — see {doc}`variants`.

`y3_buzzard/likelihood_cp.py` accepts explicit DataBlock section names for
the number counts, one-halo shear, and projected shear. `des_y3.ini` points
those options at `numcounts_fast_mass`, `shear1h_fast_mass`, and
`shear_prj_fast_mass`, so the fast-mass modules can coexist with the legacy
DES-Y1 modules without relying on DataBlock overwrites.
