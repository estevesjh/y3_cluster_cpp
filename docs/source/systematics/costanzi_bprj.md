# Correction Function $\mathcal B_{\rm prj}(R)$

`Python` + `C++` · `y3_cluster_cpp` (`src/pipelines/systematics/costanzi_bprj/`) · `Systematics` · module `costanzi_bprj` · `<1 ms/sample`

The Costanzi et al. (2026) **multiplicative correction for the optical
selection bias on the lensing profile** of the traditional max model:
instead of propagating the selection-affected bias $b_{\rm sel}(\theta)$
through the two-halo integral (the $\Sigma^{\rm prj}$ route of
{doc}`../observables/shear_projection`), the conventional
$\max(1h, b\,2h)$ profile is multiplied by a calibrated
scale-dependent factor,

$$\Sigma_{\rm corr}(R) = \mathcal B_{\rm prj}(R)\;\Sigma_{\max}(R),
\qquad
\Sigma_{\max} = \max\big(\Sigma_{1h},\, \Sigma_{2h}\big),$$

Appendix C of
[Costanzi et al. 2026, PhRvD 113, 103508](https://ui.adsabs.harvard.edu/abs/2026PhRvD.113j3508C/abstract)
(arXiv:[2604.05833](https://arxiv.org/abs/2604.05833)). The two routes
are alternative descriptions of the same physics — the excess
correlated structure along the line of sight of optically selected
clusters — and must not be combined: $\mathcal B_{\rm prj}$ only makes
sense on the max model, which carries no $b_{\rm sel}$.

## Script

- Python model + CosmoSIS module: [`src/pipelines/systematics/costanzi_bprj/python/costanzi_bprj.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/pipelines/systematics/costanzi_bprj/python/costanzi_bprj.py)
  (`CostanziBprj`, a frozen pydantic dataclass; `bprj_wall(block, R)`;
  `setup/execute/cleanup`).
- C++ twin: [`cpp/costanzi_bprj_t.hh`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/pipelines/systematics/costanzi_bprj/cpp/costanzi_bprj_t.hh)
  (`y3_cluster::CostanziBprj_t`, DataBlock constructor).
- Consumer: `likelihood_cp.py` with `shear_max_section = shear1h2h_max`
  and `is_b_proj_costanzi26 = T` ({doc}`../observables/likelihood`).
- Tests: `test/costanzi_bprj.test.{py,cc}` (golden values, values-file
  loader) and `test/likelihood_cp.test.py` (closure with $A = 0$,
  independent wall evaluation, misconfiguration errors).

## Model

A double power law with a smooth transition at the comoving cluster
radius:

$$\mathcal B_{\rm prj}(R \mid \lambda^{\rm ob}, z) =
1 + A\left(\frac{R}{R_0}\right)^{\alpha}
\left[1 + \left(\frac{R}{R_0}\right)^{\gamma}\right]^{(\beta-\alpha)/\gamma},
\qquad
R_0 = R_\lambda(\lambda^{\rm ob})\,(1+z),\quad
R_\lambda = \Big(\frac{\lambda^{\rm ob}}{100}\Big)^{0.2}\,h^{-1}{\rm Mpc}.$$

$A$ sets the amplitude of the bias, $\alpha$ and $\beta$ the inner and
outer logarithmic slopes, $R_0$ the transition scale (the redMaPPer
aperture, comoving) and $\gamma > 0$ its sharpness. $R$ must be in the
same comoving $h^{-1}$Mpc units as $R_0$. The same functional form
describes the bias on $\Delta\Sigma(R)$ with different parameter
values; both sets are exposed as class methods:

| Target | $A$ | $\alpha$ | $\beta$ | $\gamma$ | Accessor |
|---|---:|---:|---:|---:|---|
| $\Sigma(R)$ | 0.10 | 0.1 | −0.53 | 4.1 | `CostanziBprj.sigma()` |
| $\Delta\Sigma(R)$ | 0.12 | 4.11 | 0.18 | 1.82 | `CostanziBprj.dsigma()` |

The $\Sigma$ row carries the owner's 2026-09-01 specification
$\alpha = 0.1$; the arXiv version of App. C quotes $0.92$ — confirm
against the published version before sampling. The likelihood applies
the correction to the tangential shear, i.e. to $\Delta\Sigma$, so the
`dsigma` set is the one that matters for `des_y3.ini`-style runs.

The correction is evaluated on the stacked $(\text{bin}, R)$ wall in
the `NumCounts`/`Shear1h2hMax` order — redshift-major, richness
fastest, radius fastest within a bin — by `bprj_wall(block, R)`, which
reads everything but the radii from the `costanzi_bprj` DataBlock
section per sample. The four parameters may therefore be **sampled**:
put them in the values file with priors and they become nuisance
parameters of the max-model analysis.

## CosmoSIS setup

Two pieces: the module stage publishes the bin grid, the values file
supplies the parameters.

```ini
[pipeline]
modules = ... Shear1h2hMax costanzi_bprj likelihoods

[costanzi_bprj]
file = ${Y3_CLUSTER_CPP_DIR}/src/pipelines/systematics/costanzi_bprj/python/costanzi_bprj.py
lob_centers = 25.0 37.5 52.5 130.0
zob_centers = 0.275 0.425 0.575

[likelihoods]
shear_max_section    = shear1h2h_max
is_b_proj_costanzi26 = T
shear_r_perp         = <the Shear1h2hMax r_perp grid, cMpc/h>
```

```ini
; values file
[costanzi_bprj]
A     = 0.12
alpha = 4.11
beta  = 0.18
gamma = 1.82
```

- Ordering: any time before `likelihoods`; it only writes the grid.
- `lob_centers`/`zob_centers` default to the DES Y3 wall shown; a
  different section name can be passed to keep a $\Sigma$ and a
  $\Delta\Sigma$ parameter set in one pipeline.
- Requires `pydantic` in the Python environment (installed in the macOS
  env; check the NERSC env).

## Configuration options

| Option | Meaning | Units | Reference value |
|---|---|---|---|
| `lob_centers` | richness-bin centres $\lambda^{\rm ob}$ setting $R_0$ | — | 25 37.5 52.5 130 |
| `zob_centers` | redshift-bin centres setting $(1+z)$ in $R_0$ | — | 0.275 0.425 0.575 |

## DataBlock inputs

| DataBlock input | Meaning | Units / shape | Produced by |
|---|---|---|---|
| `costanzi_bprj/{A, alpha, beta, gamma}` | model parameters (sampled or fixed) | — | values file |

## DataBlock outputs

| DataBlock output | Meaning | Units / shape | Consumed by |
|---|---|---|---|
| `costanzi_bprj/lob_centers` | richness centres of the wall | `(4,)` | `likelihood_cp.py` (`bprj_wall`) |
| `costanzi_bprj/zob_centers` | redshift centres of the wall | `(3,)` | same |

Usage from code:

```python
from systematics.costanzi_bprj.python.costanzi_bprj import CostanziBprj, bprj_wall
shear_theory *= bprj_wall(block, r_perp)        # whole wall, from the datablock
bprj = CostanziBprj.from_datablock(block)       # or CostanziBprj.dsigma()
dsigma_corr = bprj(R, lob, zob) * dsigma_max    # pointwise
```

```cpp
#include "pipelines/systematics/costanzi_bprj/cpp/costanzi_bprj_t.hh"
y3_cluster::CostanziBprj_t const bprj(sample);  // [costanzi_bprj] from the values file
double const dsigma_corr = bprj(R, lob, zob) * dsigma_max;
```
