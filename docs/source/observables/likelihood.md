# Likelihood

`Python` · `y3_cluster_cpp` (`y3_buzzard/`) · `Likelihood` · module `likelihoods` · `<1 ms/sample`

Assembles the theory vector from the three observable modules, compares
it with the data vector and inverse covariance from an `.npz` file, and
writes the Gaussian $\log L$ the sampler consumes.

## Script

- Source: [`y3_buzzard/likelihood_cp.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/d7feb7504ed5dfcad84f99a1791af8a55c858aa0/y3_buzzard/likelihood_cp.py)
  (`y3_cluster_cpp` @ `d7feb75`).
- Loaded by CosmoSIS as a Python module; matched to the sampler by
  `likelihoods = likelihoods` in `[pipeline]`.

## Numerical framework

$$\gamma_t^{\rm theory}(R \mid i) =
\frac{\mathtt{shear1hmissel/vals}}{\mathrm{repeat}(\mathtt{numcountssel/vals})}
+ \mathtt{shear\_prj/vals},
\qquad
\log L = -\tfrac12 \sum_{\rm obs \in \{NC,\, Shear\}}
\delta^{\mathsf T} C^{-1} \delta,$$

with zero-guarded division (bins with $N_i \le 0$ contribute 0 to the
one-halo average). The two shear pieces add linearly because both are
tangential shear, not reduced shear — the
$1/(1 - \kappa)$ denominator was retired
({doc}`../math/index`). Shapes are hard-asserted at setup and on
every DataBlock read, so mismatches abort before an MCMC starts. In
`log_space` mode the theory is floored at $10^{-300}$ so a transient
non-positive prediction gives a large finite $\chi^2$ instead of a crash.

## CosmoSIS setup

```ini
[likelihoods]
file = ${Y3_CLUSTER_CPP_DIR}/y3_buzzard/likelihood_cp.py
filename = ${DES_CLUSTER_NERSC_DIR}/data/mock/mock_dv_buzzard.npz
verbose = F
log_space = F
```

- Ordering: last module; needs `numcountssel`, `shear1hmissel`, and
  `shear_prj` (written in this pipeline by the
  {doc}`shear_prj_frozen_physics <shear_projection>` aliases).

The likelihood derives the number of radii per bin from the length of
`data_Shear`, so both supported layouts are valid: 15 radii (180 shear
points) for the Buzzard pipeline and 10 radii (120 points) for the
widePlanck self-closure variant ({doc}`../variants`). The `des_y3.ini`
configuration also supplies the fast-mass DataBlock sections explicitly;
the legacy section names remain the defaults for the older pipeline.

## Configuration options

| Option | Meaning | Units | Reference value |
|---|---|---|---|
| `filename` | data vector + inverse covariance `.npz` (required) | — | `mock_dv_buzzard.npz` |
| `num_counts_section` | DataBlock section containing the number-count vector | — | `numcountssel` |
| `shear_1h_section` | DataBlock section containing the 1-halo shear vector | — | `shear1hmissel` |
| `shear_prj_section` | DataBlock section containing the projected shear vector | — | `shear_prj` |
| `log_space` | evaluate the Gaussian on $\ln(\rm obs)$ with the delta-method inverse covariance $C_y^{-1}[i,j] = d_i d_j C^{-1}[i,j]$; A/B-tested statistically identical to linear | — | F |
| `verbose` | per-sample $\log L$ breakdown | — | F |

Expected `.npz` keys: `data_NC (12,)`, `invcov_NC`, `data_Shear`,
`invcov_Shear` — each `invcov` either 1-D (diagonal) or 2-D (dense).
Extra keys (e.g. the embedded `fiducial_param_*` truth in the Buzzard
file) are ignored.

## DataBlock inputs

| DataBlock input | Meaning | Units / shape | Produced by |
|---|---|---|---|
| `numcountssel/vals` | $N_i[1]$ | `(12,)` | `NumCountsSel` |
| `shear1hmissel/vals` | $N_i[\gamma_t^{1h}](R)$, count-weighted | `(N_{\rm shear},)` | `Shear1hMisSel` |
| `shear_prj/vals` | $\gamma_t^{\rm prj}(R)$, per bin | `(N_{\rm shear},)` | `shear_prj_frozen_physics` (alias write) |

## DataBlock outputs

| DataBlock output | Meaning | Units / shape | Consumed by |
|---|---|---|---|
| `likelihoods/likelihoods_like` | Gaussian $\log L$ | scalar | sampler |
