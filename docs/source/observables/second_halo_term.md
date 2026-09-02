# Second Halo Term

`Python` (producer) · `y3_buzzard/halo_model_cosmosis.py` + `src/pipelines/cosmology/halo_model.py` · `Lensing` · consumed by `Shear1h2hMax` · `200–300 ms/sample` (the `compute_lensing_2h` branch)

The conventional two-halo term is the lensing signal of all the matter
that is *correlated* with the cluster but does not belong to its own
halo: neighbouring haloes and the large-scale structure they trace.
Around a halo of mass $M$ at redshift $z$ it is the halo bias times the
projected matter correlation function. In this pipeline it is
**produced** by the {doc}`halo_model <../cosmology/halo_model>` module
(behind `compute_lensing_2h`) and **consumed** by the traditional
$1h{+}2h$ max-model shear, `Shear1h2hMax` — the DES Y1 lensing model,
retained as a model option next to the reference
$1h + {\rm prj}$ composition.

In the paper's language the two-halo term is the **unselected-bias
limit** of the selection-affected two-halo term $\Sigma^{\rm prj}$ of
{doc}`shear_projection`: replace $b(M,z)\,b_{\rm sel}(\theta)$ by the
plain halo bias, drop the halo exclusion and the neighbour-by-neighbour
projection, and the $\theta$–$z$ integral collapses to a line-of-sight
projection of $b\,\xi_{\rm NL}$. The optical selection bias that
$\Sigma^{\rm prj}$ carries explicitly is then reintroduced, in the max
model, as a multiplicative correction $\mathcal B_{\rm prj}(R)$
({doc}`../systematics/costanzi_bprj`).

```{admonition} Not active in the reference pipeline
:class: important
`des_y3.ini` ({doc}`../running`) runs `halo_model` with
`compute_lensing_2h = F`: the reference shear composition is one-halo +
projection, which never reads the two-halo tables, and skipping the
Hankel-transform branch saves 200–300 ms per sample. Everything on this
page is exercised by the **`Shear1h2hMax` model option** —
`Shear1h2hMax.so`, `shear1h2h_max.py`, `Shear1h2hMaxGpu.so` under
`src/pipelines/des_y3/shear_1h2h/` — and by the likelihood's
`shear_max_section` mode ({doc}`likelihood`, {doc}`../variants`).
```

## Script

- Producer (CosmoSIS module): [`y3_buzzard/halo_model_cosmosis.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/y3_buzzard/halo_model_cosmosis.py),
  with the Hankel-transform chain in class `ct_2hTerm`
  ([`src/pipelines/cosmology/halo_model.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/pipelines/cosmology/halo_model.py),
  canonical copy; `y3_buzzard/haloModel.py` is the path-stable original),
  driving `cluster_toolkit` for $P \to \xi \to \Sigma \to \Delta\Sigma$.
- Consumers: [`src/pipelines/des_y3/shear_1h2h/cpp/0d/shear1h2h_max_t.hh`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/pipelines/des_y3/shear_1h2h/cpp/0d/shear1h2h_max_t.hh)
  (`Shear1h2hMax`, C++), [`python/0d/shear1h2h_max.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/pipelines/des_y3/shear_1h2h/python/0d/shear1h2h_max.py),
  `cuda/0d/Shear1h2hMaxGpu.cu`, and the adaptive reference
  `cpp/3d/Shear1h2hMax3d.cc`. The historical `SIG_MAX`/`KAPPA_MAX`/
  `GAMMA_MAX` models of `src/models/` compose the same max but through
  the wired `SigmaTotSel`/`DSigmaTotSel` modules, which remain broken
  ({doc}`../modules/historical`).

## The physics

### The halo–matter correlation and its projection

The mean matter density at 3-D separation $r$ from a halo of mass $M$
is $\bar\rho_m\,[1 + \xi_{\rm hm}(r \mid M, z)]$. The halo model splits
$\xi_{\rm hm}$ into the halo's own profile (the one-halo term of
{doc}`shear_halo`) and the correlated surroundings; with linear
deterministic bias the latter is

$$\xi_{\rm hm}^{2h}(r \mid M, z) = b(M, z)\;\xi_{\rm mm}(r, z),$$

where $b(M,z)$ is the Tinker-2010 halo bias ({doc}`../cosmology/halo_bias`)
and $\xi_{\rm mm}$ the matter correlation function — computed here from
the (nonlinear where available, otherwise linear) power spectrum,
$\xi_{\rm mm}(r,z) = \frac{1}{2\pi^2}\int dk\,k^2 P(k,z)\,\sin(kr)/(kr)$
({doc}`../cosmology/halo_model`, issue #9 for the linear fallback).
Projecting the excess density along the line of sight gives the
two-halo surface density,

$$\Sigma_{2h}(R \mid M, z) = \bar\rho_m \int_{-\infty}^{\infty} d\chi_\parallel\;
\xi_{\rm hm}^{2h}\!\Big(\sqrt{R^2 + \chi_\parallel^2}\,\Big|\,M, z\Big)
= b(M, z)\;\Sigma_{\rm hh}(R, z),
\qquad
\Sigma_{\rm hh}(R, z) \equiv \bar\rho_m \int d\chi_\parallel\, \xi_{\rm mm},$$

with $\bar\rho_m = \Omega_m\rho_{\rm crit,0}$ the comoving mean density
(the same reference density as the one-halo NFW tables, the unified
$\rho_m$ convention). Because the bias factors out of the projection,
the pipeline tabulates the **unbiased** $\Sigma_{\rm hh}(R,z)$ once per
sample and lets each consumer multiply by $b(M,z)$ at its own mass
nodes — one $(z, R)$ table instead of an $(M, z, R)$ cube. The lensing
observable is the excess surface density

$$\Delta\Sigma_{\rm hh}(R, z) = \bar\Sigma_{\rm hh}(<R) - \Sigma_{\rm hh}(R),
\qquad
\bar\Sigma_{\rm hh}(<R) = \frac{2}{R^2}\int_0^R s\,\Sigma_{\rm hh}(s)\,ds,$$

and the two-halo term contributes to the tangential shear as
$b\,\Delta\Sigma_{\rm hh}\,\langle\Sigma_{\rm crit}^{-1}\rangle$.

Note the differences from $\Sigma^{\rm prj}$: the projection is a
straight $\chi_\parallel$ integral at fixed transverse $R$ (no
neighbour-by-neighbour offset profile, no photo-$z$ window — the
cluster's true redshift sets $z$); the bias is the halo bias of the
*target*, not the selection-affected $b_{\rm sel}(\theta)$ of the
population; and there is no halo exclusion, so $\Sigma_{2h}$ is
unphysically large inside the halo boundary — which is exactly what the
max composition below is designed to hide.

### The max composition (DES Y1 prescription)

The one-halo and two-halo profiles are not added. Following Hayashi &
White (2008) as adopted in the DES Y1 lensing analysis
([McClintock et al. 2019, MNRAS 482, 1352](https://ui.adsabs.harvard.edu/abs/2019MNRAS.482.1352M/abstract),
arXiv:[1805.00039](https://arxiv.org/abs/1805.00039)), the total
profile is the **pointwise maximum**:

$$\Delta\Sigma_{\max}(R \mid M, z) = \max\!\Big[
\Delta\Sigma_{1h}(R \mid M, z),\;
b(M, z)\,\Delta\Sigma_{\rm hh}(R, z)\Big],$$

with $\Delta\Sigma_{1h}$ the centred + miscentred one-halo mixture of
{doc}`shear_halo`. The max is a crude but effective stand-in for halo
exclusion and the 1h–2h transition: inside the halo the steep NFW wins,
far outside the linear-bias two-halo term wins, and there is no
double-counting in between. It is nonlinear in the profiles, which has
two consequences for the pipeline: (i) the composition must be applied
*inside* the population integral, at each $(M, z)$, not to the stacked
profiles; (ii) because the two-halo term depends on $z$, the redshift
integral can no longer be contracted past the profile the way the
one-halo `0d` path does — `Shear1h2hMax` keeps a $z$-resolved
selection weight and performs a double $(\ln M, z)$ fixed-GL sum:

$$N_i[\Delta\Sigma_{\max}](R) = \int d\ln M \int dz\;
\Omega(z)\,\frac{dV}{d\Omega\,dz}\,\frac{dn}{d\ln M}(M,z)\,
S_{ij}(\ln M, z)\;\langle\Sigma_{\rm crit}^{-1}\rangle(z)\;
\Delta\Sigma_{\max}(R \mid M, z),
\qquad
\gamma_t^{\max}(R \mid i) = \frac{N_i[\Delta\Sigma_{\max}](R)}{N_i[1]}.$$

The selection-affected bias never enters this composition; the
Costanzi-2026 max-model correction multiplies the result by
$\mathcal B_{\rm prj}(R)$ instead ({doc}`../systematics/costanzi_bprj`),
and `likelihood_cp.py` applies it when `is_b_proj_costanzi26 = T`.
Compared with the reference $1h^{\rm mis} + {\rm prj}$ composition, the
max model has no $b_{\rm sel}$ boost at large $R$ and a different 1h–2h
transition; the two are compared in {doc}`../math/index`.

## Numerical recipe (`ct_2hTerm`)

Per sample, for each of the 50 redshift slices of the power-spectrum grid:

1. $\xi_{\rm mm}(r, z)$ from the per-$z$ $P(k)$ slice
   (`cluster_toolkit.xi.xi_mm_at_r`) on a fixed grid
   $r \in [10^{-3}, 10^{3}]\ {\rm cMpc}/h$ with 50 log nodes — the
   correlation must be tabulated well past the BAO scale; 50 nodes is
   the speed/accuracy sweet spot (0.1%). The tables are published with
   $b = 1$; the consumer applies $b(M,z)$.
2. $\Sigma_{\rm hh}(R, z)$ on the `r_sigma` grid via
   `cluster_toolkit.deltasigma.Sigma_at_R`, which extends the $\xi$
   table below its inner edge assuming an NFW halo
   ($M_d = 10^{13}\,M_\odot/h$, $c_d = 4$ — a numerical
   parameterisation of the extension, not physics; the inner edge is
   $10^{-3}$ cMpc/$h$, so its imprint is negligible).
3. $\Sigma \to \Delta\Sigma$ needs the interior mass $\bar\Sigma(<R)$.
   The default method (`dsigma_method='direct'`, since the issue #4
   fix) evaluates $\Sigma_{\rm hh}$ on a log grid extended down to
   $R = 10^{-3}$ cMpc/$h$ and integrates $\bar\Sigma(<R)$ by cumulative
   trapezoid — the two-halo interior mass is *integrated*, not modeled.
   Validated against closed-form NFW/Einasto chains and a converged
   fiducial anchor cross-checked by CLensPy: 0.4% max over
   $R \in [0.5, 20]$ cMpc/$h$, $z \in [0.24, 0.65]$
   (`validations/second_halo_term/` in the `scratchReports` repo).
4. The historical **NFW-sandwich stabiliser**
   (`dsigma_method='sandwich'`) — *add* an analytic $\Sigma_{\rm NFW}$
   so `DeltaSigma_at_R`'s interior extrapolation is NFW-dominated, then
   *subtract* the same halo's analytic $\Delta\Sigma_{\rm NFW}$ — stays
   selectable for comparison. With a consistent $M_d$ it is
   dummy-independent (residual $\sim$1%), but it cannot recover the
   two-halo term's own interior mass below the table edge (65% max
   deviation at small $R$ on the same benchmark), which is why `direct`
   is the default. The pre-fix code broke even the cancellation (an
   `Md/10` inconsistency) and then blanked the resulting negatives to
   NaN — 60% of the table; both defects are fixed and pinned by
   `test/halo_model.test.py`. `Shear1h2hMax` keeps a sanitize-to-0
   guard on `dSigma_hh` (exact for a max: $\max(1h, 0) = 1h$).

The consumers read `dSigma_hh(R, z)` and `bias(M, z)` through the
project's clamped `Interp2D` primitives; `Shear1h2hMax` evaluates the
one-halo mixture from the same `dSigma_nfw` table and miscentred-NFW
disk tables as `Shear1hGl`, so the two arms of the max share every
one-halo ingredient. Precision and cost of the max backends:
`0d` C++ 11 ms/sample and CUDA 8 ms at $8.3\times10^{-4}$ vs the
adaptive `Shear1h2hMax3d` reference ({doc}`shear_halo`).

## CosmoSIS setup

Enable inside the `[halo_model]` section (everything else as in the
reference — see {doc}`../cosmology/halo_model`) and add the max-model
module:

```ini
[halo_model]
compute_lensing_1h = T
compute_lensing_2h = T   ; reference run sets F

[Shear1h2hMax]
file = ${Y3_CLUSTER_CPP_DIR}/release-build/src/modules/des_y3_shear1h_0d_cpp/Shear1h2hMax.so
bin_index = 0 1 2 3 4 5 6 7 8 9 10 11
r_perp   = <radii, cMpc/h>
zt_low   = 0.05
zt_high  = 0.80
lnm_low  = 29.9336
lnm_high = 36.7300
n_lnm = 96
n_z   = 64
lob_centers = 25.0 37.5 52.5 130.0
include_miscentering = T

[likelihoods]
shear_max_section = shear1h2h_max
is_b_proj_costanzi26 = T          ; optional B_prj(R) correction
```

`Shear1h2hMax` requires `miscentering/f_mis` and `miscentering/tau_mis`
in the values file (no in-code fallback) and `haloModel/rho_m_ref`,
`bias`, `dSigma_nfw`, `dSigma_hh`. `cosmosis-models/real_pipeline_extract_max2h.ini`
is the fixture pipeline that exercises this branch.

## DataBlock outputs (when enabled)

| DataBlock output | Meaning | Units / shape | Consumed by |
|---|---|---|---|
| `haloModel/Sigma_hh` | two-halo surface density on the `r_sigma` grid, **bias not applied** (consumer multiplies by $b(M,z)$) | $M_\odot/(h\,\mathrm{pc}^2)$, `(50, 128)` | max-model diagnostics |
| `haloModel/dSigma_hh` | two-halo excess surface density, bias not applied | same, `(50, 128)` | `Shear1h2hMax`, `shear1h2h_max.py`, `Shear1h2hMaxGpu`, `Shear1h2hMax3d` |
| `shear1h2h_max/vals` | $N_i[\Delta\Sigma_{\max}\,\Sigma_{\rm crit}^{-1}](R)$, bin slow / radius fast (written by the consumer) | `(N_{\rm bins}\times N_R,)` | `likelihoods` (`shear_max_section`) |

`haloModel/Wp_hh` (the chain's internal ξ under a misleading $W_p$ name)
and its mislabeled `haloModel/Rp` axis are **no longer published**
(2026-08: unsupported; their only reader was the legacy
`wp_cluster.cuh` interpolation, which used the wrong radial axis — see
`docs/known_issues/wp_hh_rp_axis_mismatch.md`). The internal ξ stage of
the chain remains pinned by `test/halo_model.test.py`; consumers that
need ξ read `xi_nl`.
