# Richness Scaling Relation

`C++ / Python` (model, not a module) · `y3_cluster_cpp` · `Halo population`

The richness–mass relation $P(\lambda^{\rm ob} \mid M, z)$ connects halo
mass to the observed richness. It is **not a pipeline module**: the model
is evaluated inside `sel_function`, `b_sel_marg`, and `bsel`, from
parameters the sampler writes into `cluster_mor/*`. It factorises as

$$P(\lambda^{\rm ob}\mid M, z) = \int d\lambda^{\rm tr}\,
\underbrace{P(\lambda^{\rm ob}\mid\lambda^{\rm tr}, z)}_{\text{projection kernel (EMG)}}\;
\underbrace{P(\lambda^{\rm tr}\mid M, z)}_{\text{intrinsic relation (shifted Poisson)}} .$$

## Script

- Intrinsic relation (C++): [`src/models/mor_hod_t.hh`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/models/mor_hod_t.hh)
  (`MOR_HOD_t`), used by `b_sel_marg`.
- Projection kernel (C++): [`src/models/richness_kernel_t.hh`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/models/richness_kernel_t.hh)
  (EMG CDF `F_EMG`), used by the selection kernels.
- Python mirrors (line-for-line): `_p_hod_scalar` and `_cdf_lob` in
  [`src/modules/sel_function/sel_function.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/modules/sel_function/sel_function.py);
  `_p_ltr_given_M` / `_p_lob_given_ltr_emg` in
  [`y3_buzzard/bsel.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/y3_buzzard/bsel.py).
- EMG coefficient calibration: [`y3_buzzard/prj_params.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/y3_buzzard/prj_params.py)
  (`PrjParams`, the Buzzard-calibrated spline table; the standalone
  `prj_params` DataBlock module is retired).

## Numerical framework

### The Costanzi selection-effect methodology

The treatment of optical selection in this pipeline is the Costanzi
framework, developed across
[Costanzi et al. 2019a](https://ui.adsabs.harvard.edu/abs/2019MNRAS.482..490C/abstract)
(projection-effects model; arXiv:[1807.11719](https://arxiv.org/abs/1807.11719)),
[Costanzi et al. 2019b](https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.4779C/abstract)
(SDSS cluster cosmology; arXiv:[1810.09456](https://arxiv.org/abs/1810.09456)),
[Costanzi et al. 2021](https://ui.adsabs.harvard.edu/abs/2021PhRvD.103d3522C/abstract)
(DES Y1 + SPT; arXiv:[2010.13800](https://arxiv.org/abs/2010.13800)), and
[Costanzi et al. 2026](https://ui.adsabs.harvard.edu/abs/2026PhRvD.113j3508C/abstract)
(forward-modelled lensing selection bias; arXiv:[2604.05833](https://arxiv.org/abs/2604.05833)).
Its core idea: the observed richness is a noisy estimator of the halo's
true galaxy content, and the noise has a **one-sided, non-Gaussian tail
from projection effects** — line-of-sight structures contaminating the
member list — that must be modelled explicitly, because the same
contamination correlates the selection with the surrounding large-scale
structure and biases the stacked lensing signal.

**The Costanzi projection kernel.** The observed richness decomposes as
$\lambda^{\rm ob} = \lambda^{\rm tr} + \Delta^{\rm bkg} + \Delta^{\rm prj}$:
a Gaussian background/measurement fluctuation
$\Delta^{\rm bkg} \sim \mathcal{N}(\Delta\mu, \sigma^2)$ (with
$\Delta\mu < 0$ typically — redMaPPer's global background subtraction
biases $\lambda^{\rm ob}$ low), plus a one-sided projection boost
modelled as a spike at zero with an exponential tail:

$$P(\Delta^{\rm prj}\mid\lambda^{\rm tr}, z) =
(1 - f^{\rm prj})\,\delta_{\rm D}(\Delta^{\rm prj})
+ f^{\rm prj}\,\tau\, e^{-\tau\,\Delta^{\rm prj}}\,
\Theta(\Delta^{\rm prj}).$$

Convolving the two gives an **exponentially modified Gaussian** (EMG)
kernel $P(\lambda^{\rm ob}\mid\lambda^{\rm tr}, z)$. The four kernel
parameters $\{\Delta\mu, \sigma, f^{\rm prj}, \tau\}(\lambda^{\rm tr}, z)$
are calibrated empirically on synthetic-cluster injections
(Costanzi et al. 2019a); $f^{\rm prj}$ — the fraction of clusters with a
projection boost — grows with richness and redshift, and the
$f^{\rm prj} = 0$ limit recovers the pure-Gaussian "BKG" model of
Costanzi et al. 2021. In this pipeline the calibrated coefficient
splines live in `PrjParams` (`y3_buzzard/prj_params.py`).

**Closed-form bin integrals.** The selection kernels never need the EMG
pdf itself, only its CDF, which is closed-form:

$$F_{\rm EMG}(x; \mu, \sigma, \tau) =
\Phi\!\left(\frac{x-\mu}{\sigma}\right)
- \exp\!\left[-\tau(x-\mu) + \tfrac12\tau^2\sigma^2\right]
\Phi\!\left(\frac{x-\mu}{\sigma} - \tau\sigma\right),$$

so the probability of landing in richness bin
$\Delta\lambda_i$ at fixed $\lambda^{\rm tr}$ is a weighted difference
of normal and EMG CDFs at the two bin edges (evaluated via `erfcx` for
numerical stability). This is what collapses the nominally
5-dimensional number-count integral over
$(M, z^{\rm tr}, \lambda^{\rm tr}, \lambda^{\rm ob}, z^{\rm ob})$ to a
2-D $(M, z^{\rm tr})$ integral: the $\lambda^{\rm ob}$ and $z^{\rm ob}$
integrals are analytic, and the $\lambda^{\rm tr}$ integral is a fixed
Gauss–Legendre sum bracketed at
$\mu_{\rm eff} \pm L\,\sigma_{\rm eff}$ of the intrinsic relation.

**Intrinsic relation — shifted-Poisson HOD.** With
$\nu = \mu_{\rm sat} + \delta$ and $\delta = (\sigma_\lambda \mu_{\rm sat})^2$:

$$P(\lambda^{\rm tr}\mid M, z) =
\exp\!\big[-\nu + (\lambda^{\rm tr} + \delta - 1)\ln\nu
- \ln\Gamma(\lambda^{\rm tr} + \delta)\big],$$

a continuous shifted-Poisson with mean $\mu_{\rm sat}(M, z)$ set by the
HOD parameters above — Poissonian at low occupancy, super-Poissonian at
high occupancy through the halo-to-halo scatter $\sigma_\lambda$. The
exact Poisson-plus-scatter law has no closed form; Costanzi et al. 2019b
used a lookup-table skew-normal, while this continuous form
(M. Costanzi, priv. comm.) is closed-form, differentiable in $(M, z)$,
extends smoothly to the non-integer richness the quadrature needs, and
matches the exact law even in the low-$\lambda^{\rm tr}$ tail.
Numerical guards: narrow-Gaussian fallback where
$\mu_{\rm sat} \le 10^{-8}$; `z_pivot` defaults to 0.45. When both
`log10_ratio` and `log10_M1` are present, `log10_ratio` wins:
$\log_{10} M_1 = \log_{10} M_{\rm min} + \texttt{log10\_ratio}$. Two
known small divergences between evaluators: `b_sel_marg` and `bsel` use
the satellite term without the central-galaxy shift $\lambda_{\rm cen}$,
and `bsel` defaults its pivot to 0.4544. (The log-normal
mass–richness model of Costanzi et al. 2021 is the drop-in alternative;
only the quadrature bracket changes.)

**Beyond the counts.** The same kernel is what sources the lensing
selection bias: the projection boost $\Delta^{\rm prj}$ is produced by
the very line-of-sight structure that adds two-halo lensing signal, so
Costanzi et al. 2026 propagate $f^{\rm prj}$, $\tau$, and the richness
excess $\lambda^{\rm ob} - \lambda^{\rm tr}$ into the scale-dependent
selection-affected bias $b_{\rm sel}(\theta)$ used by
{doc}`../systematics/bsel` and {doc}`../observables/shear_projection`.

Where the reference pipeline evaluates it:

- {doc}`sel_function <../systematics/sel_function>` builds
  $S_{ij}(\ln M, z)$ from both pieces — this is how it enters
  {doc}`NumCountsSel <../observables/number_counts>` and
  {doc}`Shear1hMisSel <../observables/shear_halo>`;
- {doc}`b_sel_marg <../systematics/bsel>` weights the $P[X]$ operators by
  the shifted-Poisson pdf;
- {doc}`bsel <../systematics/bsel>` uses the EMG pdf in the
  $\lambda^{\rm tr}$ marginalisation.

Full derivation (log-normal alternative, EMG CDF closed form, quadrature
placement): {doc}`../math/index`.

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

