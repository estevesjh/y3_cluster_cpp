# Second Halo Term

`Python` (producer) · `y3_buzzard/halo_model_cosmosis.py` + `src/pipelines/cosmology/halo_model.py` · `Lensing` ·
consumed by `Shear1h2hMax` · `200–300 ms/sample` (the `compute_lensing_2h` branch)

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
projection of $b\,\xi_{\rm NL}$. The optional tagged Bsel correction
({doc}`../systematics/bsel_profile`) is applied to the max-model
$\Delta\Sigma$ at every $(M, z)$ node when `Shear1h2hMax` is configured
with `bsel = T`.

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

- Producer (CosmoSIS module):
  [`y3_buzzard/halo_model_cosmosis.py`][halo_cosmosis_src],
  with the Hankel-transform chain in class `ct_2hTerm`
  ([`src/pipelines/cosmology/halo_model.py`][halo_model_src],
  canonical copy; `y3_buzzard/haloModel.py` is the path-stable original),
  driving `cluster_toolkit` for $P \to \xi \to \Sigma \to \Delta\Sigma$.
- Consumers:
  [`src/pipelines/des_y3/shear_1h2h/cpp/0d/shear1h2h_max_t.hh`][shear_max_t_src]
  (`Shear1h2hMax`, C++),
  [`python/0d/shear1h2h_max.py`][shear_max_py_src],
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

#### The 3D halo-exclusion piecewise max model

The one-halo and two-halo profiles are not combined via a naive 2D pointwise maximum or sum. Instead, the model is
physically formulated in 3D real space with halo exclusion and piecewise approximation:

1. In 3D real space, halo exclusion is defined on the correlation function:

$$\xi_{\rm hm}(r \mid M, z) = \max\!\Big[\xi_{1h}(r \mid M, z),\; b(M, z)\,\xi_{\rm mm}(r, z)\Big],$$

with transition radius $r_{\rm trans}$ defined by $\xi_{1h}(r_{\rm trans}) = b(M, z)\,\xi_{\rm mm}(r_{\rm trans}, z)$.

2. The projected excess surface density $\Delta\Sigma_{\max}$ is approximated around transition radius and is assembled
   by building a piecewise function. Because $\Delta\Sigma(R) = \bar{\Sigma}(<R) - \Sigma(R)$ and
   $\frac{d}{dR}[R^2\bar{\Sigma}(<R)] = 2R\,\Sigma(R)$, enforcing the quadratic matching for $\Delta\Sigma$ uniquely
   determines the exact unified profiles for **$\Sigma(R)$**, **$\bar{\Sigma}(<R)$**, and **$\Delta\Sigma(R)$**:

#### Projected Surface Density $\Sigma(R)$
Outside the core ($R \ge r_{\rm trans}$), sightlines never pierce $r < r_{\rm trans}$, so the local surface density is
purely 2-halo: $\Sigma(R) = b(M, z)\,\Sigma_{\rm hh}(R, z)$. Inside the core ($R < r_{\rm trans}$), matching the
quadratic expansion yields:

$$\Sigma_{\rm hm}(R, M, z) = \begin{cases} 
\Sigma_{1h}(R, M) + \Delta_{\rm gap}^{(\Sigma)}(M, z) + 2\,\Delta_{\rm gap}(M, z)\left[1
  - \left(\dfrac{R}{r_{\rm trans}(M, z)}\right)^2\right] & \text{if } R < r_{\rm trans}(M, z) \\[10pt] 
b(M, z)\,\Sigma_{\rm hh}(R, z) & \text{if } R \ge r_{\rm trans}(M, z) 
\end{cases}$$

where the boundary surface density gap is:
$$\Delta_{\rm gap}^{(\Sigma)}(M, z) \equiv b(M, z)\,\Sigma_{\rm hh}\big(r_{\rm trans}(M, z), z\big)
  - \Sigma_{1h}\big(r_{\rm trans}(M, z), M\big).$$

#### Enclosed Mean Surface Density $\bar{\Sigma}(<R)$
$$\bar{\Sigma}_{\rm hm}(<R, M, z) = \begin{cases} 
\bar{\Sigma}_{1h}(<R, M) + \Delta_{\rm gap}^{(\Sigma)}(M, z) + 2\,\Delta_{\rm gap}(M, z)
  - \Delta_{\rm gap}(M, z)\left(\dfrac{R}{r_{\rm trans}(M, z)}\right)^2 & \text{if } R < r_{\rm trans}(M, z) \\[10pt] 
b(M, z)\,\bar{\Sigma}_{\rm hh}(<R, z) + \dfrac{\mathcal{C}_0(M, z)}{R^2} \times 10^{-12}
  & \text{if } R \ge r_{\rm trans}(M, z) 
\end{cases}$$

#### Excess Surface Density $\Delta\Sigma(R) = \bar{\Sigma}(<R) - \Sigma(R)$
Subtracting $\Sigma(R)$ from $\bar{\Sigma}(<R)$, the constant sheet $\Delta_{\rm gap}^{(\Sigma)} + 2\Delta_{\rm gap}$
cancels out identically, leaving:

$$\Delta\Sigma_{\rm hm}(R, M, z) = \begin{cases} 
\Delta\Sigma_{1h}(R, M) + \Delta_{\rm gap}(M, z)\left(\dfrac{R}{r_{\rm trans}(M, z)}\right)^2
  & \text{if } R < r_{\rm trans}(M, z) \\[10pt] 
b(M, z)\,\Delta\Sigma_{\rm hh}(R, z) + \dfrac{\mathcal{C}_0(M, z)}{R^2} \times 10^{-12}
  & \text{if } R \ge r_{\rm trans}(M, z) 
\end{cases}$$

where $\mathcal{C}_0 = \Delta M_{\rm 3D}(r_{\rm trans}) / \pi$ (in $M_\odot/h$) and
$$\Delta_{\rm gap}(M, z) = \left[b(M, z)\,\Delta\Sigma_{\rm hh}(r_{\rm trans}, z) +
\frac{\mathcal{C}_0}{r_{\rm trans}^2}\times 10^{-12}\right] - \Delta\Sigma_{1h}(r_{\rm trans}, M),$$
with $\Delta\Sigma_{1h}$ the **centred** NFW profile (`haloModel/dSigma_nfw`); the max model carries
no miscentering term.

* **Exact Identity:** $\Delta\Sigma_{\rm hm}(R) \equiv \bar{\Sigma}_{\rm hm}(<R) - \Sigma_{\rm hm}(R)$ holds
  analytically everywhere.
* **Continuity:** All three profiles ($\Sigma$, $\bar{\Sigma}$, $\Delta\Sigma$) are strictly $C^0$ continuous at
  $R = r_{\rm trans}$.
* **DataBlock inputs:** All tables are strictly 2D arrays: $\Sigma_{1h}(R, M)$, $\Sigma_{\rm hh}(R, z)$,
  $r_{\rm trans}(M, z)$, $\mathcal{C}_0(M, z)$, $\Delta_{\rm gap}(M, z)$, and $\Delta_{\rm gap}^{(\Sigma)}(M, z)$.
  `r_trans`, `c0_trans` and `delta_gap_1` are **required** (published by `halo_model` when
  `compute_lensing_1h = compute_lensing_2h = T`); there is no $\max(1h, 2h)$ fallback.

This piecewise form is `profile_method = piecewise_approx`, the default:
a fast and precise approximation of the max model.

#### Full integral (`profile_method = gauss_legendre`)

When the full integral is needed, `Shear1h2hMax` projects
$\xi_{\max}(r) = \max[\xi_{\rm nfw}(r \mid M),\, b(M,z)\,\xi_{\rm nl}(r,z)]$
directly on Gauss–Legendre nodes (`OperatorMax1h2h_gl`,
`src/pipelines/cosmology/operator_max1h2h_gl.{hh,py}`):

$$\Sigma(R) = 2\rho_m\int_0^{l_{\max}} \xi_{\max}\big(\sqrt{R^2 + l^2}\big)\,dl,
\qquad
\bar\Sigma(<R) = \frac{2}{R^2}\int_0^R R'\,\Sigma(R')\,dR',
\qquad
\Delta\Sigma = \bar\Sigma(<R) - \Sigma(R),$$

with `n_los` (default 64) line-of-sight and `n_aperture` (default 32)
aperture nodes. Here $\xi_{\rm nfw} = \rho_{\rm NFW}/\rho_m - 1$ (M200m,
from `haloModel/concentration` and `haloModel/rho_m_ref`) and
$\xi_{\rm nl}$ is read from `xi_nl/{r, z, xi_nl}` with a bicubic spline;
the piecewise tables are not needed. Measured on the real
`des_y3_cpp0d_fast_apriori.ini` block (12 bins × 10 radii, one sample,
Apple M5):

| `profile_method` (`n_los`/`n_aperture`) | C++ | Python | Precision |
|---|---:|---:|---|
| `piecewise_approx` | 0.003 s | 14 ms | Agrees with `gauss_legendre` to $\le 0.43\%$ across all radii |
| `gauss_legendre` 64/32 | 0.12 s (~40×) | 0.35 s | 5.8e-5 vs 128/64 |
| `gauss_legendre` 128/64 | 0.41 s (~140×) | — | 1.7e-5 vs 256/128 |
| `gauss_legendre` 256/128 | 1.5 s | — | 8.9e-6 vs 512/256 |

Bilinear $\xi_{\rm nl}$ interpolation stalled the node convergence at
~4e-5 and biased the stack by 6e-4, which is why the GL path reads it
bicubically (`Interp2D(..., bicubic)` / `Bilinear2D(..., cubic=True)`).
Python and C++ agree to ~5e-15 (`piecewise_approx`) and ~1e-7
(`gauss_legendre`, scipy vs GSL bicubic).

The transition and piecewise terms depend on $M$ and $z$, which has two consequences for the pipeline: (i) the
composition must be applied *inside* the population integral, at each $(M, z)$, not to the stacked profiles; (ii)
because the two-halo term depends on $z$, the redshift integral can no longer be contracted past the profile the way
the one-halo `0d` path does — `Shear1h2hMax` keeps a $z$-resolved selection weight and performs a double $(\ln M, z)$
fixed-GL sum:

$$N_i[\Delta\Sigma_{\max}](R) = \int d\ln M \int dz\;
\Omega(z)\,\frac{dV}{d\Omega\,dz}\,\frac{dn}{d\ln M}(M,z)\,
S_{ij}(\ln M, z)\;\langle\Sigma_{\rm crit}^{-1}\rangle(z)\;
\Delta\Sigma_{\max}(R \mid M, z),
\qquad
\gamma_t^{\max}(R \mid i) = \frac{N_i[\Delta\Sigma_{\max}](R)}{N_i[1]}.$$

The selection-affected bias never enters this composition directly; when a
selection-bias correction is needed, set `bsel = T` and the tagged Bsel
operator $\Delta\Sigma_{\rm sel} = \mathcal{B}\,\Delta\Sigma - M_{\rm sel}/(\pi R^2)$
is applied at each $(M, z)$ node ({doc}`../systematics/bsel_profile`).
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
centred one-halo arm from the same `dSigma_nfw` table as `Shear1hGl`
(no miscentering). Precision and cost of the two `profile_method`
options are in the table above. The CUDA `Shear1h2hMaxGpu` and the
adaptive `Shear1h2hMax3d` are separate backends that still implement the
older miscentred piecewise model ({doc}`shear_halo`).

## CosmoSIS setup

Enable inside the `[halo_model]` section (everything else as in the
reference — see {doc}`../cosmology/halo_model`) and add the max-model
module after its upstream modules:

```ini
[pipeline]
modules = consistency GrowthFactor cp_camb MfTinker halo_model
          average_sigma_crit_inv sel_function
          Shear1h2hMax

[halo_model]
compute_lensing_1h = T
; reference run sets compute_lensing_2h = F
compute_lensing_2h = T

[Shear1h2hMax]
file = ${Y3_CLUSTER_CPP_DIR}/release-build/src/modules/des_y3_shear1h_0d_cpp/Shear1h2hMax.so
bin_index = 0 1 2 3 4 5 6 7 8 9 10 11
r_perp   = 0.20000 0.28599 0.40896 0.58480 0.83625 1.19581 1.70998 2.44521 3.49658 5.00000
zt_low   = 0.05
zt_high  = 0.80
lnm_low  = 29.9336
lnm_high = 36.7300
n_lnm = 96
n_z   = 64
lob_centers = 25.0 37.5 52.5 130.0
bsel = F
; bsel_model = Costanzi26          ; or Sunayama23; required if bsel = T
profile_method = piecewise_approx  ; or gauss_legendre
n_los = 64                         ; gauss_legendre only
n_aperture = 32                    ; gauss_legendre only

[likelihoods]
shear_max_section = shear1h2h_max
```

The Python twin takes the same options under `[shear1h2h_max]` with
`file = ${Y3_CLUSTER_CPP_DIR}/src/pipelines/des_y3/shear_1h2h/python/0d/shear1h2h_max.py`.
Both write `shear1h2h_max/vals`, so run one or the other, never both.
`cosmosis-models/des_y3_cpp0d_fast_apriori.ini` and
`cosmosis-models/des_y3_python0d_fast_apriori.ini` are worked examples.

Inputs: always `haloModel/{r_sigma, lnM, z, bias}`,
`average_sigma_crit_inv`, `sel_function/S_stack` and the cosmology, HMF
and distance sections. `piecewise_approx` adds
`haloModel/{dSigma_nfw, dSigma_hh, r_trans, c0_trans, delta_gap_1}`
(and, with `bsel = T`, `Sigma_bar_nfw`, `Sigma_bar_hh`, `delta_gap_0`);
`gauss_legendre` adds `haloModel/{concentration, rho_m_ref}` and
`xi_nl/{r, z, xi_nl}`. `Shear1h2hMax` reads no `miscentering` section.

A single-point check with the test sampler (macOS: first
`source ~/cosmosis_y3/cosmosis_init_macos.sh`, {doc}`../building_macos`):

```bash
cosmosis cosmosis-models/des_y3_cpp0d_fast_apriori.ini -p runtime.sampler=test test.save_dir=out \
  "pipeline.modules=consistency GrowthFactor cp_camb MfTinker halo_model \
    average_sigma_crit_inv sel_function Shear1h2hMax" \
  Shear1h2hMax.profile_method=gauss_legendre
```

writes `out/shear1h2h_max/vals.txt`. The full option reference is in
`src/pipelines/des_y3/shear_1h2h/README.md` ("How to run
`Shear1h2hMax`"); Bsel configuration is in
{doc}`../systematics/bsel_profile`.

## DataBlock outputs (when enabled)

| DataBlock output | Meaning | Units / shape | Consumed by |
|---|---|---|---|
| `haloModel/Sigma_hh` | 2h surface density, unscaled by $b$ | $M_\odot/(h\,\mathrm{pc}^2)$, `(50, 128)` | Diagnostics |
| `haloModel/dSigma_hh` | 2h excess surface density, unscaled by $b$ | Same, `(50, 128)` | `Shear1h2hMax` backends |
| `shear1h2h_max/vals` | $N_i[\Delta\Sigma_{\max}\,\Sigma_{\rm crit}^{-1}](R)$ | `(N_{\rm bin} N_R,)` | `likelihoods` |

`haloModel/Wp_hh` (the chain's internal ξ under a misleading $W_p$ name)
and its mislabeled `haloModel/Rp` axis are **no longer published**
(2026-08: unsupported; their only reader was the legacy
`wp_cluster.cuh` interpolation, which used the wrong radial axis). The
internal ξ stage of the chain remains pinned by
`test/halo_model.test.py`; consumers that need ξ read `xi_nl`.

[halo_cosmosis_src]:
  https://github.com/estevesjh/y3_cluster_cpp/blob/master/y3_buzzard/halo_model_cosmosis.py
[halo_model_src]:
  https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/pipelines/cosmology/halo_model.py
[shear_max_t_src]:
  https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/pipelines/des_y3/shear_1h2h/cpp/0d/shear1h2h_max_t.hh
[shear_max_py_src]:
  https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/pipelines/des_y3/shear_1h2h/python/0d/shear1h2h_max.py
