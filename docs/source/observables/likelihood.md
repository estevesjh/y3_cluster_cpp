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

```{warning}
**Shape contract, 120 vs 180.** As committed, `likelihood_cp.py`
hard-codes `_SHEAR_N_R = 10` (→ `_SHEAR_N = 120`), while the Buzzard ini
and `mock_dv_buzzard.npz` use **15 radii → 180 shear points**. With the
committed constants, `setup()` rejects the Buzzard data vector
(`data_Shear has size 180, expected 120`). Running the Buzzard pipeline
requires `_SHEAR_N_R = 15` (it also controls the `np.repeat` tiling of
the number counts). The 120-point contract matches the widePlanck
self-closure variant ({doc}`../variants`).
```

## Configuration options

| Option | Meaning | Units | Reference value |
|---|---|---|---|
| `filename` | data vector + inverse covariance `.npz` (required) | — | `mock_dv_buzzard.npz` |
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

