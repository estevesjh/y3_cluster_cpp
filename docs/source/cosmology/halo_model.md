# Halo Model

`Python` · `y3_cluster_cpp` (`y3_buzzard/`) · `Halo population` · module `halo_model` · `141 ms/sample`

Publishes the halo-model ingredients consumed by half the downstream
pipeline: the Tinker-2010 halo bias $b(M,z)$, the nonlinear matter
correlation function $\xi_{\rm NL}(r,z)$, and the NFW one-halo lensing
tables $\Sigma_{\rm NFW}$, $\Delta\Sigma_{\rm NFW}$ read by
`Shear1hMisSel`.

## Script

- Source: [`y3_buzzard/halo_model_cosmosis.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/d7feb7504ed5dfcad84f99a1791af8a55c858aa0/y3_buzzard/halo_model_cosmosis.py)
  (`y3_cluster_cpp` @ `d7feb75`).
- Helpers: `y3_buzzard/haloModel.py` (classes `biasModel`, `lensingModel`,
  `ct_2hTerm`, `scaleShiftCosmo`) and `y3_buzzard/nfwModel.py` (analytic
  NFW $\Sigma/\Delta\Sigma$); numerical backend `cluster_toolkit`.
- Loaded by CosmoSIS as a Python module.

## Numerical framework

The single most multi-product module in the pipeline: one `execute`
call computes up to **four families of quantities**, all backed by
`cluster_toolkit` (a hard runtime dependency — see the install note in
[BUILDING.md](https://github.com/estevesjh/y3_cluster_cpp/blob/master/BUILDING.md)).
The governing expressions, in the order they are evaluated:

**1. Halo bias** — Tinker 2010 at the growth-rescaled $z{=}0$ peak
height ($\Delta = 200\bar\rho_m$):

$$b(M, z) = b_{\rm Tinker}\!\left(\frac{\nu(M)}{D(z)/D(0)}\right),
\qquad \nu(M) = \frac{\delta_c}{\sigma(M, z{=}0)} .$$

**2. Nonlinear matter correlation** — per redshift slice, the Fourier
transform of the (nonlinear, or fallback linear) power spectrum:

$$\xi_{\rm NL}(r, z) = \frac{1}{2\pi^2}\int dk\, k^2\,
P_{\rm NL}(k, z)\, \frac{\sin kr}{kr},
\qquad r \in [10^{-3}, 10^{3}]\ {\rm cMpc}/h,\ 128\ \text{nodes}.$$

**3. One-halo NFW lensing tables** (`compute_lensing_1h`) — the
analytic Wright & Brainerd projected NFW,

$$\Sigma_{\rm NFW}(R \mid M, c(M)), \qquad
\Delta\Sigma_{\rm NFW}(R \mid M, c(M)) =
\bar\Sigma(<R) - \Sigma(R),$$

with the Child-18 concentration–mass relation $c(M)$, on the
$(M, R_\perp)$ grid — the centred profile `Shear1hMisSel` consumes.

**4. Two-halo lensing tables** (`compute_lensing_2h`, **off** in the
reference run) — the Hankel chain
$P \to \xi \to \Sigma_{\rm hh} \to \Delta\Sigma_{\rm hh}$
({doc}`../observables/second_halo_term`).

The algorithm per sample: build the mass grid; compute $\nu(M)$ once at
$z = 0$ (`cluster_toolkit.peak_height.nu_at_M` on the linear $P(k)$);
loop over the 50 power-spectrum redshift slices evaluating
`bias_at_nu(nu / (D(z)/D(0)))` and `ct.xi.xi_mm_at_r`; write the bias
and $\xi_{\rm NL}$ grids plus $\rho_c(z) = \Omega_m\rho_{c,0}(1+z)^3$
and the `scaleShiftCosmo` factors; then run whichever lensing branches
are enabled. GSL's abort-on-error handler is disabled at import (via
ctypes) so extreme cosmologies raise Python exceptions instead of
killing the MCMC worker.

Two load-bearing details:

- **The $D(0)$ renormalisation**: CosmoSIS growth is
  matter-domination-normalised ($D(0) \simeq 0.76$); dividing by $D(z)$
  un-normalised inflates $\nu$ by $1/D(0) \simeq 1.32$ and the bias by
  up to $2\times$ (a pre-May-2026 bug, fixed).
- **The 1h/2h flag split**: the 2h Hankel loop costs 200–300 ms/sample
  at $N_z = 50$ — the majority of the module's runtime — and nothing in
  the reference pipeline reads its outputs, so `compute_lensing_2h = F`
  is a pure skip.

Model details: {doc}`halo_bias` (bias), {doc}`../observables/second_halo_term`
(lensing branches); algorithm source:
[pipeline_modules.tex](https://github.com/estevesjh/y3_cluster_cpp/blob/master/docs/pipeline_modules.tex)
§halo_model.

## CosmoSIS setup

```ini
[halo_model]
file = ${Y3_CLUSTER_CPP_DIR}/y3_buzzard/halo_model_cosmosis.py
R_perp_min = 0.05
R_perp_max = 10.0
R_perp_bins = 128
Radii_min = 1.0
Radii_max = 35.0
Radii_bins = 128
M_min = 1.0e12
M_max = 1.0e16
M_bins = 100
compute_lensing_1h = T
compute_lensing_2h = F
```

- Requires `Y3_CLUSTER_CPP_DIR` and a Python environment with
  `cluster_toolkit`.
- Ordering: after `cp_camb` (power spectra) and `GrowthFactor`; before
  `Shear1hMisSel`, `b_sel_marg`, `bsel`, `shear_prj_frozen_physics`.
- `compute_lensing_2h = F` in the reference run: nothing in this pipeline
  reads the two-halo tables, and skipping the Hankel-transform branch saves
  200–300 ms per sample. The branch itself is documented in
  {doc}`../observables/second_halo_term`.

## Configuration options

| Option | Meaning | Units | Reference value |
|---|---|---|---|
| `R_perp_min/max/bins` | projected-radius grid of the 1h lensing tables | cMpc/$h$ | 0.05, 10.0, 128 |
| `Radii_min/max/bins` | radius grid of the 2h $W_p$ tables | cMpc/$h$ | 1.0, 35.0, 128 |
| `M_min/max/bins` | halo-mass grid | $M_\odot/h$ | $10^{12}$, $10^{16}$, 100 |
| `compute_lensing_1h` | publish NFW $\Sigma/\Delta\Sigma$ tables (needed by `Shear1hMisSel`) | — | T |
| `compute_lensing_2h` | publish two-halo `Sigma_hh/dSigma_hh/Wp_hh` tables | — | F |

## DataBlock inputs

| DataBlock input | Meaning | Units / shape | Produced by |
|---|---|---|---|
| `cosmological_parameters/{omega_m, omega_b, h0}` | background cosmology | scalars | `consistency` |
| `matter_power_lin/{k_h, p_k, z}` | linear power (peak height $\nu(M)$) | $h/\mathrm{Mpc}$, $(\mathrm{Mpc}/h)^3$ | `cp_camb` |
| `matter_power_nl/*` | nonlinear power for $\xi_{\rm NL}$ (falls back to linear if absent) | same | `cp_camb` (optional) |
| `growth_parameters/{z, d_z}` | growth factor for the $\nu$ redshift scaling | `(406,)` | `GrowthFactor` |

## DataBlock outputs

| DataBlock output | Meaning | Units / shape | Consumed by |
|---|---|---|---|
| `haloModel/lnM`, `m_h` | mass grid | $\ln M_\odot/h$, `(100,)` | `b_sel_marg`, `shear_prj_frozen_physics`, `bsel`, `Shear1hMisSel` |
| `haloModel/z` | redshift grid | `(50,)` | same |
| `haloModel/bias` | Tinker-2010 halo bias $b(M,z)$ | `(50, 100)` | `b_sel_marg`, `shear_prj_frozen_physics`, `bsel` — see {doc}`halo_bias` |
| `haloModel/rhoc` | critical density $\rho_c(z)$ | `(50,)` | diagnostics |
| `xi_nl/{r, z, xi_nl}` | nonlinear matter correlation function | $r$: `(128,)` cMpc/$h$; `(50, 128)` | `b_sel_marg`, `shear_prj_frozen_physics` |
| `haloModel/{r_sigma, Sigma_nfw, dSigma_nfw, concentration, scale_shift, hubble_shift, k}` | NFW 1h lensing tables (`compute_lensing_1h = T`) | `r_sigma`: `(128,)` cMpc/$h$; tables `(100, 128)` | `Shear1hMisSel` |
| `haloModel/{Rp, Wp_hh, Sigma_hh, dSigma_hh}` | two-halo lensing tables — **not written** in the reference run (`compute_lensing_2h = F`) | `(128,)`, `(100, 128)` | {doc}`../observables/second_halo_term` variants only |

