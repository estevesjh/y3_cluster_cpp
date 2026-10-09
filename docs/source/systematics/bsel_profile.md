# Bsel profiles for `Shear1h2hMax`

`Python` + `C++` · `Shear1h2hMax` · optional 3-D-envelope selection-bias
profiles

This page documents the tagged Bsel profiles that apply the selection-bias
model to the raw surface-density profile before the max-model observable is
formed. They are separate from the existing angular `[bsel]` module and from
the legacy `costanzi_bprj` post-processing correction.

The feature is opt-in. It requires the 3-D-envelope path because the profile
calculation needs the underlying surface-density information. The ordinary
`Shear1h2hMax` path and its output section remain unchanged when Bsel is off.

## Tagged implementations

| Tag | Python kernel | C++ profile | Max-model consumer |
|---|---|---|---|
| `Costanzi26` | `src/pipelines/systematics/BselCostanzi26.py` (`BselCostanzi26`) | `BselCostanzi26` in `src/systematics/bsel_profile.hh` | `Shear1h2hMaxSelCostanzi26` |
| `Sunayama23` | `src/pipelines/systematics/BselSunayama23.py` (`BselSunayama23`) | `BselSunayama23` in `src/systematics/bsel_profile.hh` | `Shear1h2hMaxSelSunayama23` |

`BselModels` in `bsel_profile.py` and `bsel_profile.hh` is the shared
tag dispatcher. It accepts exactly `Costanzi26` or `Sunayama23`; the two
scientific prescriptions remain separate after dispatch.

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

## CosmoSIS configuration

The `[Shear1h2hMax]` section selects the profile and its numerical settings.
The `bsel_section` option points to the values section containing the
calibration for the selected tag.

```ini
[Shear1h2hMax]
file = ${Y3_CLUSTER_CPP_DIR}/release-build/src/modules/des_y3_shear1h_0d_cpp/Shear1h2hMax.so
bin_index = 0 1 2 3 4 5 6 7 8 9 10 11
r_perp = 0.20000 0.28599 0.40896 0.58480 0.83625 1.19581 1.70998 2.44521 3.49658 5.00000
zt_low = 0.05
zt_high = 0.80
lnm_low = 29.9336
lnm_high = 36.7300

three_d_envelope = T
bsel = T
bsel_model = Costanzi26
bsel_section = bsel_profile_costanzi26
bsel_n_gl = 64
```

The corresponding values section for the Costanzi tag is:

```ini
[bsel_profile_costanzi26]
; Raw-Sigma calibration from Costanzi et al. (2026), App. C.
A = 0.10
alpha = 0.92
beta = -0.53
gamma = 4.1
```

For a Sunayama23 run, change only the selected tag and section:

```ini
[Shear1h2hMax]
three_d_envelope = T
bsel = T
bsel_model = Sunayama23
bsel_section = bsel_profile_sunayama23
bsel_n_gl = 64

[bsel_profile_sunayama23]
; One Pi0 and R0 value per richness-bin row; c is shared.
lambda_bin = 0 1 2 3
pi0 = <Pi0_bin0> <Pi0_bin1> <Pi0_bin2> <Pi0_bin3>
r0 = <R0_bin0> <R0_bin1> <R0_bin2> <R0_bin3>
c = <shared_c>
```

The Sunayama values above are a configuration template, not calibrated
defaults. The final values must match the approved richness-bin wall and the
units used by the lensing profile.

The existing `[bsel]` section must not be renamed or reused: it belongs to
the angular selection-bias module. Likewise, `costanzi_bprj` should not be
enabled for the same `Shear1h2hMax` run, because it represents an alternative
selection-bias treatment.

## Options

| Option | Meaning | Required when |
|---|---|---|
| `three_d_envelope` | Enables the raw-profile path needed by Bsel | Always `T` when `bsel = T` |
| `bsel` | Enables the tagged Bsel profile | Optional; defaults to `F` |
| `bsel_model` | Model tag: `Costanzi26` or `Sunayama23` | `bsel = T` |
| `bsel_section` | DataBlock section containing the selected calibration | `bsel = T` |
| `bsel_n_gl` | Gauss–Legendre resolution for the profile correction | `bsel = T` |

Configuration errors should be explicit: an unknown tag, a missing values
section, incompatible wall lengths, or `bsel = T` without
`three_d_envelope = T` must stop the module rather than silently falling back
to the unselected profile.

## API map

### Python

- `BselCostanzi26.__call__`, `derivative`, `from_source`, and `validate_wall`.
- `BselSunayama23.__call__`, `derivative`, `from_source`, and `validate_wall`.
- `BselModels.from_source`, `__call__`, and `derivative` for explicit tag
  dispatch.
- `Shear1h2hMaxSelCostanzi26` and `Shear1h2hMaxSelSunayama23` as the tagged
  selected-profile consumers.

### C++

- `y3_cluster::BselCostanzi26` and `y3_cluster::BselSunayama23` for the two
  kernels.
- `y3_cluster::BselModels` for explicit tag parsing and dispatch.
- `Shear1h2hMaxSelCostanzi26` and `Shear1h2hMaxSelSunayama23` for the selected
  profile consumers.
- `Shear1h2hMax` remains the CosmoSIS module entry point and writes the
  existing `shear1h2h_max/vals` output.

See {doc}`../observables/second_halo_term` for the max-model context and
{doc}`costanzi_bprj` for the legacy multiplicative correction.
