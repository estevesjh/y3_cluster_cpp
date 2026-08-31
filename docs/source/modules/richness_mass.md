# Richness Scaling Relation

The richness–mass relation $P(\lambda^{\rm ob} \mid M, z)$ connects halo
mass to the observed richness. This is a model, not a pipeline module —
it is evaluated inside `sel_function`, `b_sel_marg`, and `bsel`, from
parameters the sampler writes into `cluster_mor/*`. It factorises as

$$P(\lambda^{\rm ob}\mid M, z) = \int d\lambda^{\rm tr}\,
\underbrace{P(\lambda^{\rm ob}\mid\lambda^{\rm tr}, z)}_{\text{projection kernel (EMG)}}\;
\underbrace{P(\lambda^{\rm tr}\mid M, z)}_{\text{intrinsic relation (shifted Poisson)}} .$$


## Numerical framework

### The Costanzi selection-effect methodology

The treatment of optical selection in this pipeline is the Costanzi
framework (Costanzi et al. 2019a,b, 2021, 2026) — full references in
{doc}`../math/index`, §"Selection functions".

Its core idea: the observed richness is a noisy estimator of the halo's
true galaxy content, and the noise has a **one-sided, non-Gaussian tail
from projection effects** — line-of-sight structures contaminating the
member list — that must be modelled explicitly, because the same
contamination correlates the selection with the surrounding large-scale
structure and biases the stacked lensing signal.

**The two pieces, briefly.** The projection kernel
$P(\lambda^{\rm ob}\mid\lambda^{\rm tr}, z)$ is an exponentially
modified Gaussian (EMG): a Gaussian background/measurement fluctuation
plus a one-sided exponential projection tail, with four kernel
parameters $\{\Delta\mu, \sigma, f^{\rm prj}, \tau\}(\lambda^{\rm tr}, z)$
calibrated on synthetic-cluster injections (Costanzi et al. 2019a); the
calibrated coefficient splines live in `PrjParams`
(`y3_buzzard/prj_params.py`). The intrinsic relation
$P(\lambda^{\rm tr}\mid M, z)$ is a continuous shifted-Poisson HOD
(M. Costanzi, priv. comm.), closed-form and differentiable in $(M, z)$,
replacing the lookup-table skew-normal of Costanzi et al. 2019b. Because
the EMG's CDF is closed-form, the selection kernels never need the pdf
itself — this is what collapses the nominally 5-D number-count integral
over $(M, z^{\rm tr}, \lambda^{\rm tr}, \lambda^{\rm ob}, z^{\rm ob})$
to a 2-D $(M, z^{\rm tr})$ integral. Full equations, closed-form CDFs,
and the log-normal alternative (Costanzi et al. 2021): {doc}`../math/index`,
§"Selection functions".

**Numerical guards and known divergences.** Narrow-Gaussian fallback
where $\mu_{\rm sat} \le 10^{-8}$; `z_pivot` defaults to 0.45. When both
`log10_ratio` and `log10_M1` are present, `log10_ratio` wins:
`log10_M1 = log10_Mmin + log10_ratio`. Two known small divergences between evaluators: `b_sel_marg` and `bsel` use
the satellite term without the central-galaxy shift $\lambda_{\rm cen}$,
and `bsel` defaults its pivot to 0.4544.

**Beyond the counts.** The same kernel is what sources the lensing
selection bias: the projection boost $\Delta^{\rm prj}$ is produced by
the very line-of-sight structure that adds two-halo lensing signal, so
Costanzi et al. 2026 propagate $f^{\rm prj}$, $\tau$, and the richness
excess $\lambda^{\rm ob} - \lambda^{\rm tr}$ into the scale-dependent
selection-affected bias $b_{\rm sel}(\theta)$ used by
{doc}`../selection/bsel` and {doc}`../observables/shear_projection`.

Where the reference pipeline evaluates it:

- {doc}`sel_function <../selection/sel_function>` builds
  $S_{ij}(\ln M, z)$ from both pieces — this is how it enters
  {doc}`NumCountsSel <../observables/number_counts>` and
  {doc}`Shear1hMisSel <../observables/shear_halo>`;
- {doc}`b_sel_marg <../selection/bsel>` weights the $P[X]$ operators by
  the shifted-Poisson pdf;
- {doc}`bsel <../selection/bsel>` uses the EMG pdf in the
  $\lambda^{\rm tr}$ marginalisation.
  
## Script

- Intrinsic relation (C++): [`src/models/mor_hod_t.hh`](https://github.com/estevesjh/y3_cluster_cpp/blob/d7feb7504ed5dfcad84f99a1791af8a55c858aa0/src/models/mor_hod_t.hh)
  (`MOR_HOD_t`), used by `b_sel_marg`.
- Projection kernel (C++): [`src/models/richness_kernel_t.hh`](https://github.com/estevesjh/y3_cluster_cpp/blob/d7feb7504ed5dfcad84f99a1791af8a55c858aa0/src/models/richness_kernel_t.hh)
  (EMG CDF `F_EMG`), used by the selection kernels.
- Python mirrors (line-for-line): `_p_hod_scalar` and `_cdf_lob` in
  [`src/modules/sel_function/sel_function.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/d7feb7504ed5dfcad84f99a1791af8a55c858aa0/src/modules/sel_function/sel_function.py);
  `_p_ltr_given_M` / `_p_lob_given_ltr_emg` in
  [`y3_buzzard/bsel.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/d7feb7504ed5dfcad84f99a1791af8a55c858aa0/y3_buzzard/bsel.py).
- EMG coefficient calibration: [`y3_buzzard/prj_params.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/d7feb7504ed5dfcad84f99a1791af8a55c858aa0/y3_buzzard/prj_params.py)
  (`PrjParams`, the Buzzard-calibrated spline table; the standalone
  `prj_params` DataBlock module is retired).


## Parameters (DataBlock)

Sampled parameters, written by the values file into `cluster_mor`:

| DataBlock key | Meaning | Units | Read by |
|---|---|---|---|
| `cluster_mor/log10_Mmin` | mass where $\mu_{\rm sat}$ turns on | $\log_{10} M_\odot/h$ | `sel_function`, `b_sel_marg`, `bsel` |
| `cluster_mor/log10_M1` or `log10_ratio` | satellite normalisation mass (or its ratio to `Mmin`) | $\log_{10} M_\odot/h$ | same |
| `cluster_mor/alpha` | satellite power-law slope | — | same |
| `cluster_mor/epsilon` | redshift evolution exponent | — | same |
| `cluster_mor/sigma_lambda` | intrinsic scatter parameter | — | same |
| `cluster_mor/z_pivot` | redshift pivot (optional) | — | same |

