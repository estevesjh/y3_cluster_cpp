# Likelihood

`Python` · `y3_cluster_cpp` (`src/pipelines/buzzard/likelihoods/`) · `Likelihood` · module `likelihoods` · `<1 ms/sample`

Assembles the theory vector from the observable modules, compares it
with the data vector and inverse covariance from an `.npz` file, and
writes the Gaussian $\log L$ the sampler consumes.

## Script

- Source: [`src/pipelines/buzzard/likelihoods/likelihood_cp.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/pipelines/buzzard/likelihoods/likelihood_cp.py)
  (moved from `y3_buzzard/likelihood_cp.py`; sibling-repo inis that
  still point at the old path must be updated).
- Loaded by CosmoSIS as a Python module; matched to the sampler by
  `likelihoods = likelihoods` in `[pipeline]`.
- Test: `test/likelihood_cp.test.py` (`likelihood_cp_test`) — closure of
  both shear modes and the $\mathcal B_{\rm prj}$ option on a synthetic
  DataBlock.

## Numerical framework

Default (reference) composition, one-halo + projection:

$$\gamma_t^{\rm theory}(R \mid i) =
\frac{\mathtt{shear1h\_gl/vals}}{\mathrm{repeat}(\mathtt{numcounts\_sij\_gl/vals})}
+ \mathtt{shear\_prj\_gl/cl},
\qquad
\log L = -\tfrac12 \sum_{\rm obs \in \{NC,\, Shear\}}
\delta^{\mathsf T} C^{-1} \delta,$$

with zero-guarded division (bins with $N_i \le 0$ contribute 0 to the
one-halo average). The projection piece is the **clustered channel
`cl`** — $\Sigma^{\rm prj}$, the correlated excess a
random-point-subtracted measurement contains; the mean-field `rnd`
channel must integrate to zero for a differential $\Delta\Sigma$ and is
not added (verified: `cl` converges to $b\,\bar\rho_m\,\xi$ at large $R$
while `rnd + cl` diverges). The module falls back to `vals` only if the
projection module publishes no `cl`. The two shear pieces add linearly
because both are tangential shear, not reduced shear — the
$1/(1 - \kappa)$ denominator was retired ({doc}`../math/index`).

Max-model composition (`shear_max_section` set):

$$\gamma_t^{\rm theory}(R \mid i) =
\frac{\mathtt{shear1h2h\_max/vals}}{\mathrm{repeat}(N_i)}
\;\times\;\big[\mathcal B_{\rm prj}(R \mid \lambda_i, z_j)\big]_{\texttt{is\_b\_proj\_costanzi26}},$$

no projection term ({doc}`second_halo_term`). The optional
Costanzi-2026 correction $\mathcal B_{\rm prj}$ is evaluated per sample
by `bprj_wall(block, shear_r_perp)` from the `costanzi_bprj` DataBlock
section ({doc}`../systematics/costanzi_bprj`); requesting it without a
max section is a configuration error, since the $1h + {\rm prj}$ path
already carries the selection bias through $b_{\rm sel}$.

Two further multiplicative options act on either shear theory after
composition: a $\rho_m(z)$ density-evolution factor $(1+z_j)^p$ per
redshift bin (`shear_1pz_power`, default 0; absorbs the comoving-vs-
physical surface-density tilt against the Buzzard data vector, issue
#22), and a radial **scale cut** (`shear_r_min`/`shear_r_max`) that
drops radii from data, theory, and covariance — the dense inverse
covariance is cut by inverting, slicing, and re-inverting, so removed
radii are marginalised, not conditioned.

Shapes are hard-asserted at setup and on every DataBlock read, so
mismatches abort before an MCMC starts. In `log_space` mode the Gaussian
is evaluated on $\ln({\rm obs})$ with the delta-method inverse
covariance $C_y^{-1}[i,j] = d_i d_j C^{-1}[i,j]$ linearised about the
fixed data vector (constant across the chain), and the theory is
floored at $10^{-300}$ so a transient non-positive prediction gives a
large finite $\chi^2$ instead of a crash. The linear-vs-log A/B
({doc}`../variants`) found statistically identical posteriors, so
`log_space = F` is the default.

## CosmoSIS setup

```ini
[likelihoods]
file = ${Y3_CLUSTER_CPP_DIR}/src/pipelines/buzzard/likelihoods/likelihood_cp.py
filename = ${DES_CLUSTER_NERSC_DIR}/data/mock/mock_dv_buzzard.npz
num_counts_section = numcounts_sij_gl
shear_1h_section   = shear1h_gl
shear_prj_section  = shear_prj_gl
verbose = F
log_space = F
```

- Ordering: last module; needs the three observable sections named
  above (the defaults are the DES Y1 names `numcountssel`,
  `shear1hmissel`, `shear_prj`, so the DES Y1 pipeline runs unchanged).
- The number of radii per bin is derived from the length of
  `data_Shear`, so both layouts are valid: 15 radii (180 shear points)
  for the Buzzard pipeline and 10 radii (120 points) for the widePlanck
  self-closure variant ({doc}`../variants`).

## Configuration options

| Option | Meaning | Units | Reference value |
|---|---|---|---|
| `filename` | data vector + inverse covariance `.npz` (required) | — | `mock_dv_buzzard.npz` |
| `num_counts_section` | DataBlock section of the number-count vector | — | `numcounts_sij_gl` (default `numcountssel`) |
| `shear_1h_section` | section of the $N_i$-weighted one-halo shear | — | `shear1h_gl` (default `shear1hmissel`) |
| `shear_prj_section` | section of the projected shear; its `cl` key is read | — | `shear_prj_gl` (default `shear_prj`) |
| `shear_max_section` | if set, switch to the max model: theory = `<section>/vals` / $N_i$, no projection term | — | unset (`shear1h2h_max` to enable) |
| `is_b_proj_costanzi26` | multiply the max-model theory by $\mathcal B_{\rm prj}(R)$ from the `costanzi_bprj` section | — | F |
| `shear_r_perp` | comoving radii of the shear wall (needed by the scale cut and $\mathcal B_{\rm prj}$) | cMpc/$h$ | the 10-radius widePlanck grid |
| `shear_r_min`, `shear_r_max` | keep only `shear_r_min ≤ R ≤ shear_r_max` (marginalising the rest) | cMpc/$h$ | 0, ∞ (no cut) |
| `shear_1pz_power` | multiply bin $j$ of the shear theory by $(1+z_j)^p$ | — | 0 |
| `shear_zbin_reps` | representative $z_j$ per redshift bin for that factor | — | `0.275 0.435 0.575` |
| `log_space` | Gaussian on $\ln(\rm obs)$ with the delta-method inverse covariance | — | F |
| `verbose` | per-sample $\log L$ breakdown | — | F |

Expected `.npz` keys: `data_NC (12,)`, `invcov_NC`, `data_Shear`,
`invcov_Shear` — each `invcov` either 1-D (diagonal) or 2-D (dense).
Extra keys (e.g. the embedded `fiducial_param_*` truth in the Buzzard
file) are ignored.

## DataBlock inputs

| DataBlock input | Meaning | Units / shape | Produced by |
|---|---|---|---|
| `numcounts_sij_gl/vals` | $N_i[1]$ | `(12,)` | `NumCountsSijGl` |
| `shear1h_gl/vals` | $N_i[\gamma_t^{1h}](R)$, count-weighted | `(N_{\rm shear},)` | `Shear1hGl` |
| `shear_prj_gl/cl` (fallback `vals`) | $\gamma_t^{\rm prj}(R)$, clustered channel, per bin | `(N_{\rm shear},)` | `ShearPrjGl` |
| `shear1h2h_max/vals` | $N_i[\gamma_t^{\max}](R)$ — max-model mode only | `(N_{\rm shear},)` | `Shear1h2hMax` |
| `costanzi_bprj/{A, alpha, beta, gamma, lob_centers, zob_centers}` | $\mathcal B_{\rm prj}$ parameters and bin grid — `is_b_proj_costanzi26` only | scalars, `(4,)`, `(3,)` | values file + `costanzi_bprj` module |

## DataBlock outputs

| DataBlock output | Meaning | Units / shape | Consumed by |
|---|---|---|---|
| `likelihoods/likelihoods_like` | Gaussian $\log L$ | scalar | sampler |

The boost factor $B(R)$ published by {doc}`../systematics/boost_factor`
is **not** read by this likelihood: source dilution is a division of
the combined theory shear that has not yet been wired into
`likelihood_cp.py` (see that page).
