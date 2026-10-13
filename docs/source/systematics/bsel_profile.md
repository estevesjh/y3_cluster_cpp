# Optical selection-bias corrections for `Shear1h2hMax`

`Python` + `C++` · `Shear1h2hMax` · optical selection-bias correction
functions

This page documents the tagged optical selection-bias corrections applied to
the raw surface-density profile before the max-model observable is formed.
They are separate from the existing angular `[bsel]` module.

The feature is opt-in. The ordinary `Shear1h2hMax` path and its output
section remain unchanged when the correction is off.

The pull-request documentation preview is built from the same branch as the
code changes, so this page can be checked before the pull request is merged.

## Tagged implementations

| Tag | Python kernel | C++ kernel | Values section |
|---|---|---|---|
| `Costanzi26` | `BselCostanzi26.py` (`BselCostanzi26`) | `BselCostanzi26` in `Bsel.hh` | `boost_selection_costanzi26` |
| `Sunayama23` | `BselSunayama23.py` (`BselSunayama23`) | `BselSunayama23` in `Bsel.hh` | `boost_selection_sunayama23` |

The legacy section names `bsel_profile_costanzi26` and
`bsel_profile_sunayama23` are still accepted.

`BselModels` in `selection_boost/Bsel.py` and `selection_boost/Bsel.hh` is the shared
tag dispatcher. It accepts exactly `Costanzi26` or `Sunayama23`; the two
scientific prescriptions remain separate after dispatch.

The Costanzi26 correction is the smooth form from Costanzi et al. (2026),
Appendix C, Eq. (23):

$$
\mathcal{B}_{\rm sel}(R) = 1 + A x^\alpha
\left(1+x^\gamma\right)^{(\beta-\alpha)/\gamma},
\qquad x = R/R_0,
\qquad R_0 = R_\lambda(\lambda_{\rm ob})(1+z).
$$

The Sunayama23 correction uses the piecewise contribution from Sunayama et
al. (2023), Eq. (28), and exposes the multiplicative factor used by the
profile calculation:

$$
\Pi(R) =
\begin{cases}
\Pi_0 R/R_0, & R \le R_0,\\
\Pi_0 + c\ln(R/R_0), & R > R_0,
\end{cases}
\qquad
\mathcal{B}_{\rm sel}(R) = 1+\Pi(R).
$$

The `Sunayama23` kernel returns the multiplicative factor `1 + Pi(R)`. Its
piecewise `Pi(R)` contribution is zero at the origin, rises linearly to
`Pi0` at `R0`, and then follows the shared logarithmic-slope branch. The
derivative is unchanged by the leading constant one.

```{figure} ../_static/img/shear1h2h_max_bsel_comparison.png
:alt: Comparison of Shear1h2hMax with no Bsel, Costanzi26, and Sunayama23
:width: 100%

Synthetic `Shear1h2hMax` diagnostic showing the no-Bsel profile, the direct
Python Costanzi26 Bsel application with a 50% range in `A`, and the direct
Python Sunayama23 Bsel application using the supplied min/mean/max values.
The model curves use the same Sanzo Wada
plate as CLensPy, `vol1-114`: Orange Yellow for Costanzi26 and Antwarp Blue
for Sunayama23, with black for the no-Bsel baseline and each uncertainty band
matching its corresponding mean curve.
```

## The Bsel operator on $\Delta\Sigma$

$\mathcal{B}_{\rm sel}(R)$ multiplies the surface density $\Sigma$, so
the excess surface density picks up a non-local term. For each
$(\ln M, z)$ node of the `Shear1h2hMax` sum,

$$
\Delta\Sigma_{\rm sel}(R) = \mathcal{B}_{\rm sel}(R)\,\Delta\Sigma(R)
- \frac{M_{\rm sel}(R)}{\pi R^2},
\qquad
\frac{M_{\rm sel}(R)}{\pi R^2} = R\int_0^1 u^2\,
\bar\Sigma(<uR)\,\mathcal{B}_{\rm sel}'(uR)\,du .
$$

How it is evaluated depends on the `Shear1h2hMax` `profile_method`
({doc}`../observables/second_halo_term`):

- `piecewise_approx`: `BselDeltaSigmaOperator` (`Bsel.py` / `Bsel.hh`)
  integrates over $u$ with `n_gl = 20` Gauss–Legendre nodes, using
  $\bar\Sigma(<r)$ built from `haloModel/{Sigma_bar_nfw, Sigma_bar_hh,
  delta_gap_0}`. The Sunayama23 derivative jumps at $R_0$
  ($\Pi_0/R_0 \to c/R$), so the integral is split at $u = R_0/R$
  (`BselModels.break_radius`: $R_0$ for Sunayama23, 0 for the smooth
  Costanzi26).
- `gauss_legendre`: the equivalent local form
  $\Delta\Sigma_{\rm sel} = \frac{2}{R^2}\int_0^R R'\,\mathcal{B}\,\Sigma\,dR'
  - \mathcal{B}(R)\,\Sigma(R)$, i.e. $\mathcal{B}$ weights the aperture
  nodes of the full integral; no extra tables.

Measured on the real `des_y3_cpp0d_fast_apriori.ini` block (12 bins ×
10 radii), `piecewise_approx` agrees with `gauss_legendre` to ≤ 0.41%
with Costanzi26 and to 0.56% at $R = 5$ Mpc/$h$ with Sunayama23 ($\Pi_0 = 0.45$,
$R_0 = 3.5$, $c = -0.1$; 2.9% before the split at $R_0$). Python and C++
agree to ~5e-15 (`piecewise_approx`) and ~1e-7 (`gauss_legendre`).

## CosmoSIS configuration

The `[Shear1h2hMax]` section selects the profile and its numerical settings.
The `bsel_model` tag determines the standard values section for that model.

```ini
[Shear1h2hMax]
file = ${Y3_CLUSTER_CPP_DIR}/release-build/src/modules/des_y3_shear1h_0d_cpp/Shear1h2hMax.so
bin_index = 0 1 2 3 4 5 6 7 8 9 10 11
r_perp = 0.20000 0.28599 0.40896 0.58480 0.83625 1.19581 1.70998 2.44521 3.49658 5.00000
zt_low = 0.05
zt_high = 0.80
lnm_low = 29.9336
lnm_high = 36.7300
profile_method = piecewise_approx   ; or gauss_legendre

bsel = T
bsel_model = Costanzi26
```

The Python module takes the same options under `[shear1h2h_max]`
(`file = ${Y3_CLUSTER_CPP_DIR}/src/pipelines/des_y3/shear_1h2h/python/0d/shear1h2h_max.py`).
Run one or the other, never both: they write the same
`shear1h2h_max/vals`.

The corresponding values section for the Costanzi tag is:

```ini
[boost_selection_costanzi26]
; Raw-Sigma calibration from Costanzi et al. (2026), App. C.
A = 0.03
alpha = 0.92
beta = -0.53
gamma = 4.1
```

For a Sunayama23 run, change only the selected tag and section:

```ini
[Shear1h2hMax]
bsel = T
bsel_model = Sunayama23

[boost_selection_sunayama23]
; Scalars apply to every richness bin; c is shared.
pi0 = 0.45
r0 = 3.5
c = -0.1
; Per-bin rows instead:
; lambda_bin = 0 1 2 3
; pi0 = <Pi0_bin0> <Pi0_bin1> <Pi0_bin2> <Pi0_bin3>
; r0 = <R0_bin0> <R0_bin1> <R0_bin2> <R0_bin3>
```

The Sunayama values above are an example, not calibrated
defaults. The final values must match the approved richness-bin wall and the
units used by the lensing profile.

The existing `[bsel]` section must not be renamed or reused: it belongs to
the angular selection-bias module.

### Quick run

With the CosmoSIS environment loaded (`source ~/cosmosis_y3/cosmosis_init_macos.sh`
on macOS, {doc}`../building_macos`), the test sampler runs one point.
`-p` overrides module options; `-v` sets (or adds) values-file entries:

```bash
cosmosis cosmosis-models/des_y3_cpp0d_fast_apriori.ini -p runtime.sampler=test test.save_dir=out \
  "pipeline.modules=consistency GrowthFactor cp_camb MfTinker halo_model \
    average_sigma_crit_inv sel_function Shear1h2hMax" \
  Shear1h2hMax.bsel=T Shear1h2hMax.bsel_model=Sunayama23 \
  -v boost_selection_sunayama23.pi0=0.45 boost_selection_sunayama23.r0=3.5 boost_selection_sunayama23.c=-0.1
```

`[halo_model]` must set `compute_lensing_1h = T` and
`compute_lensing_2h = T` (as `des_y3_cpp0d_fast_apriori.ini` does). The
result is in `out/shear1h2h_max/vals.txt`.

## Options

| Option | Meaning | Required when |
|---|---|---|
| `bsel` | Enables the tagged Bsel profile | Optional; defaults to `F` |
| `bsel_model` | Model tag: `Costanzi26` or `Sunayama23` | `bsel = T` |
| `lob_centers` | Richness centres for Costanzi26 scale $R_\lambda(\lambda_{\rm ob})$ | Defaults to `25 37.5 52.5 130` |

Configuration errors should be explicit: an unknown tag, a missing
model-specific values section, or incompatible wall lengths must stop the module rather than
silently falling back to the unselected profile.

## API map

### Python

- `BselCostanzi26.__call__`, `derivative`, `from_source`, and `validate_wall`.
- `BselSunayama23.__call__`, `derivative`, `from_source`, and `validate_wall`.
- `BselModels.from_source`, `__call__`, `derivative`, and `break_radius`
  for explicit tag dispatch.
- `BselDeltaSigmaOperator.apply(delta_sigma, mean_sigma_fn, R, ...)` — the
  non-local $\Delta\Sigma$ operator used by `Shear1h2hMax`.
- `Shear1h2hMaxSelCostanzi26` / `Shear1h2hMaxSelSunayama23` — multiply-only
  helpers kept in the library and its tests; `Shear1h2hMax` no longer
  uses them (they omit the $M_{\rm sel}/(\pi R^2)$ term).

### C++

- `y3_cluster::BselCostanzi26` and `y3_cluster::BselSunayama23` for the two
  kernels.
- `y3_cluster::BselModels` for explicit tag parsing, dispatch, and
  `break_radius`.
- `y3_cluster::BselDeltaSigmaOperator` for the non-local $\Delta\Sigma$
  operator (`n_gl = 20`, split at the break radius).
- `Shear1h2hMaxSelCostanzi26` / `Shear1h2hMaxSelSunayama23` and
  `make_shear1h2h_max_selected_profile` — multiply-only helpers, not used
  by `Shear1h2hMax`.
- `Shear1h2hMax` is the CosmoSIS module entry point and writes
  `shear1h2h_max/vals`.

See {doc}`../observables/second_halo_term` for the max-model context.

## References

- Costanzi et al. (2026), Appendix C, [arXiv:2604.05833](https://arxiv.org/abs/2604.05833).
- Sunayama et al. (2023), Section 4.2, [arXiv:2309.13025](https://arxiv.org/abs/2309.13025).
