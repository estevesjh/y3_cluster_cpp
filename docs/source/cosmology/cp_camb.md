# Linear Power Spectrum Emulator

`Python` · `y3_cluster_cpp` (emulators from `camb-emulator`) · `Cosmology` · module `cp_camb` · `4 ms/sample`

Replaces the CAMB Boltzmann call with a CosmoPower neural-network
emulator of the linear matter power spectrum, and publishes background
distances via astropy — no downstream module needs CAMB at all. One
emulator forward pass per sample.

## Script

- CosmoSIS interface: [`src/modules/cp_camb/cp_camb.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/d7feb7504ed5dfcad84f99a1791af8a55c858aa0/src/modules/cp_camb/cp_camb.py)
  (`y3_cluster_cpp` @ `d7feb75`), loaded by CosmoSIS as a Python module.
- **The emulator code itself lives in the external
  [estevesjh/camb-emulator](https://github.com/estevesjh/camb-emulator)
  repository** (active pipeline under
  [`camb-for-cp/`](https://github.com/estevesjh/camb-emulator/tree/42f20382e619161b214add3961ce5c7a325b4401/camb-for-cp)
  @ `42f2038`):
  - [`cp_numpy.py`](https://github.com/estevesjh/camb-emulator/blob/42f20382e619161b214add3961ce5c7a325b4401/camb-for-cp/cp_numpy.py)
    — the numpy-only inference wrapper `cp_camb.py` imports at runtime
    (no TensorFlow dependency inside CosmoSIS);
  - `models/` — the trained CosmoPower artifacts (`.pkl` + exported
    `.npz`, produced by
    [`export_cosmopower_numpy.py`](https://github.com/estevesjh/camb-emulator/blob/42f20382e619161b214add3961ce5c7a325b4401/camb-for-cp/export_cosmopower_numpy.py));
  - `configs/`, `scripts/`, `slurm/` — the CAMB training-set generation
    (Latin-hypercube sampling, CAMB runs, cleaning) and GPU training;
  - [`PIPELINE.md`](https://github.com/estevesjh/camb-emulator/blob/42f20382e619161b214add3961ce5c7a325b4401/camb-for-cp/PIPELINE.md)
    — the step-by-step instructions to run CAMB and retrain the
    emulators on Perlmutter, end to end.

## Numerical framework

The emulators are trained at $z = 0$ only; redshift evolution is
reconstructed with the linear growth factor:

$$P(k, z) = \left[\frac{D(z)}{D(0)}\right]^2 P_{\rm emu}(k, z{=}0),
\qquad
P_{\rm emu} = 10^{\,\mathtt{NN}(h_0,\, \Omega_m,\, \Omega_b,\, n_s,\, \sigma_8,\, m_\nu)}.$$

Per sample: (1) validate the parameter vector against the emulator's
trained box (`parameters_min/max`) and reject *before* any GSL-backed
downstream code sees the draw ($\Omega_b \ge \Omega_m$ also rejects);
(2) write astropy distances; (3) one NN forward pass per loaded emulator,
growth-broadcast, `put_grid`. All loaded emulators are checked at setup
to share one $k$ grid. The reference values file fixes `mnu = 0` because
the $D(z)^2$ rescaling assumes scale-independent growth.

Two independent networks per artifact generation: total-matter
$P^{\rm mm}(k)$ (→ `matter_power_lin`) and CDM+baryon $P^{\rm cb}(k)$
(→ `cdm_baryon_power_lin` for `MfTinker`). Emulator-vs-CAMB accuracy on
the held-out test set (~20k cosmologies × 506 $k$-modes, v2c numbers
from the repo README; the reference run loads the v3c artifacts):
median $|P_{\rm pred}/P_{\rm true} - 1| \approx 0.07\%$, 99th
percentile $\lesssim 1.5\%$, on both networks. Downstream validation —
residuals propagated to the number counts —
in [emulator_validation.tex](https://github.com/estevesjh/y3_cluster_cpp/blob/master/docs/emulator_validation.tex)
and `camb-for-cp/hmf_report/`.

## CosmoSIS setup

```ini
[cp_camb]
file = ${Y3_CLUSTER_CPP_DIR}/src/modules/cp_camb/cp_camb.py
emulator_repo      = /pscratch/sd/j/jesteves/github/camb-emulator/camb-for-cp
linear_pk_path      = /pscratch/sd/j/jesteves/github/camb-emulator/camb-for-cp/models/camb_linear_v3c_emulator.npz
linear_nonu_pk_path = /pscratch/sd/j/jesteves/github/camb-emulator/camb-for-cp/models/camb_linear_nonu_v3c_emulator.npz
zmin = 0.0
zmax = 4.0
nz = 50
```

- Ordering: after `consistency` and `GrowthFactor` (it reads
  `growth_parameters`); before `MfTinker` and `halo_model`.
- `zmax = 4.0` so `average_sigma_crit_inv` can cover the source $p(z)$
  tail.

## Configuration options

| Option | Meaning | Units | Reference value |
|---|---|---|---|
| `emulator_repo` | path prepended to `sys.path` so `cp_numpy` imports | — | `camb-for-cp` checkout |
| `linear_pk_path` | total-matter linear emulator (required) | — | `camb_linear_v3c_emulator.npz` |
| `linear_nonu_pk_path` | CDM+baryon (no-neutrino) emulator | — | `camb_linear_nonu_v3c_emulator.npz` |
| `nonlinear_pk_path` | optional nonlinear emulator | — | unset |
| `nonu_fallback` | copy the linear grid into `cdm_baryon_power_lin` when no nonu emulator is given | — | F (default) |
| `zmin`, `zmax`, `nz` | output redshift grid | — | 0.0, 4.0, 50 |
| `write_distances` | publish `distances/*` via astropy `FlatLambdaCDM` | — | T (default) |
| `apply_growth` | rescale $P(k, 0)$ by $D(z)^2/D(0)^2$ | — | auto-on ($z{=}0$-only emulators) |

## DataBlock inputs

| DataBlock input | Meaning | Units / shape | Produced by |
|---|---|---|---|
| `cosmological_parameters/{h0, omega_m, omega_b, n_s, sigma8, mnu}` | the 6 emulator inputs — the amplitude parameter is $\sigma_8$, not $\ln 10^{10}A_s$ | `mnu` in eV | sampler / `consistency` |
| `growth_parameters/{z, d_z}` | growth factor for the redshift reconstruction (renormalised by $D(0)$) | `(406,)` | `GrowthFactor` |

## DataBlock outputs

| DataBlock output | Meaning | Units / shape | Consumed by |
|---|---|---|---|
| `matter_power_lin/{z, k_h, p_k}` | total-matter linear $P(k,z)$ | $h/\mathrm{Mpc}$, $(\mathrm{Mpc}/h)^3$; `(50, 506)` | `halo_model` |
| `cdm_baryon_power_lin/{z, k_h, p_k}` | CDM+baryon linear $P(k,z)$ | same grid | `MfTinker` (`matter_power_lin_version = 2`) |
| `matter_power_nl/{z, k_h, p_k}` | nonlinear $P(k,z)$ — only if a nonlinear emulator is loaded | same | `halo_model` (optional) |
| `distances/{z, a, d_a, d_m, d_l, d_c, h, mu, nz}` | astropy background distances; Mpc (CAMB convention, no $h$); `d_c = d_m` (flat); `h = H(z)/c` in Mpc⁻¹ | `(50,)` each | `average_sigma_crit_inv`, `NumCountsSel`, `Shear1hMisSel`, `b_sel_marg`, `bsel`, `shear_prj_frozen_physics` |
| `cosmological_parameters/cp_camb_invalid_reason` | rejection reason string (only on rejected draws, with module status 1 → $\log L = -\infty$) | — | diagnostics |


```{todo}
**Non-linear power spectrum not yet emulated.** The module supports a
`nonlinear_pk_path` emulator, but none is trained: `matter_power_nl` is
not published in the reference run, and `halo_model` falls back to the
linear spectrum for $\xi_{\rm NL}$. A halofit-level non-linear emulator
still needs to be trained in `camb-emulator` and wired in here.
```
