# Systematics implementations

This directory owns the selection-systematics pieces used by the DES Y3
observable pipelines. It is the canonical maintained location for these
models; the older copies under `src/pipelines/shared/`,
`src/pipelines/cosmology/`, and `src/models/` remain available as
compatibility references.

```text
systematics/
├── selection_richness/python/
│   ├── sel_function.py       # S_ij(ln M, z)
│   └── sel_kernels.py        # offline/reference selection helpers
├── selection_bias/
│   ├── python/bsel.py        # b_small, b_large from P1, I1, J
│   └── cpp/bsel_bins_t.hh    # exact wall-row lookup for C++ consumers
├── selection_function/python/
│   └── prj_params.py         # Costanzi projection-kernel coefficients
├── selection_boost/
│   ├── Bsel.py                # tagged dispatcher and profile consumers
│   ├── BselCostanzi26.py     # Costanzi26 B_sel(R) kernel
│   ├── BselSunayama23.py     # Sunayama23 B_sel(R) kernel
│   └── Bsel.hh                # C++ kernels and profile consumers
└── shear_prj/cpp/
    ├── sigma_prj_t.hh                # exact-z projection shear core
    ├── sigma_prj_frozen_t.hh         # frozen-physics projection backend
    └── sigma_prj_frozen_interp_t.hh  # continuous frozen Cuhre backend
```

The DES Y3 pipeline imports the Python selection modules from here. Its
projection-shear C++ driver includes the `shear_prj` core from here. The
lower-level numerical and DataBlock utilities remain in the sibling
`../shared/` package; halo-model physics remains in `../cosmology/`.

## Tagged optical selection boost: `Bsel(R)`

The `selection_boost/` implementation provides the optical selection-bias
factor applied to the local max-model profile by `Shear1h2hMax`. It is
enabled in the module configuration with `bsel = T` and selected with the
exact tag `Costanzi26` or `Sunayama23`.

The selected profile is applied after the local max operation,

```text
DeltaSigma_max(R) = max(DeltaSigma_1h(R), b(M,z) DeltaSigma_2h(R))
DeltaSigma_selected(R) = B_sel(R) DeltaSigma_max(R)
```

The existing `shear1h2h_max/vals` output section and bin/radius wall are
preserved. The Python and C++ `Shear1h2hMax` implementations use the same
tagged consumer boundary.

### Costanzi26

`BselCostanzi26` implements Costanzi et al. (2026), Appendix C, Eq. (23):

```text
B_sel(R) = 1 + A x^alpha [1 + x^gamma]^((beta - alpha)/gamma)
x = R / R0
R0 = R_lambda(lambda_ob) (1 + z)
```

`R` and `R0` use the comoving `h^-1 Mpc` convention. The four parameters
are scalar values read from the standard section
`[boost_selection_costanzi26]`:

```ini
[Shear1h2hMax]
bsel = T
bsel_model = Costanzi26

[boost_selection_costanzi26]
A = 0.10
alpha = 0.92
beta = -0.53
gamma = 4.1
```

The Python class is
`systematics.selection_boost.BselCostanzi26.BselCostanzi26`; its public
methods are:

- `__call__(R, lob, z)` — evaluate `B_sel(R)`;
- `derivative(R, lob, z)` — evaluate the analytic radial derivative;
- `r0(lob, z)` — return the Costanzi transition scale;
- `from_source(source, section=...)` — read the four parameters from a
  DataBlock-like source;
- `validate_wall(lob, z)` — validate richness and redshift wall vectors.

The C++ twin is `y3_cluster::BselCostanzi26` in
`selection_boost/Bsel.hh`. It provides the same call and derivative
interface, plus a DataBlock constructor.

### Sunayama23

`BselSunayama23` implements the piecewise contribution from Sunayama et al.
(2023), Eq. (28), and returns the multiplicative factor including the
leading one:

```text
Pi(R) = Pi0 R / R0,                 R <= R0
        Pi0 + c log(R / R0),        R > R0
B_sel(R) = 1 + Pi(R)
```

The model is calibrated by richness-bin row. `pi0` and `r0` are vectors,
`c` is shared by all rows, and `lambda_bin` gives the corresponding bin
labels. If `lambda_bin` is omitted, positional labels `0, 1, ...` are used.
The standard DataBlock section is `[bsel_profile_sunayama23]`:

```ini
[Shear1h2hMax]
bsel = T
bsel_model = Sunayama23

[bsel_profile_sunayama23]
lambda_bin = 0 1 2 3
pi0 = 0.20 0.45 0.45 0.45
r0 = 2.0 3.5 3.5 3.5
c = -0.10
```

The values above are an interface example. Production values must be filled
with the approved richness-bin calibration. `pi0` is dimensionless, `r0`
uses the same comoving `h^-1 Mpc` convention as `R`, and `c` is the shared
dimensionless logarithmic slope.

The Python class is
`systematics.selection_boost.BselSunayama23.BselSunayama23`; its public
methods are:

- `__call__(R, bin_index)` — evaluate `1 + Pi(R)`;
- `derivative(R, bin_index)` — evaluate the branch derivative;
- `from_source(source, section=...)` — read the calibration vectors and
  shared slope from a DataBlock-like source;
- `validate_wall(bin_index)` — verify that requested bin labels exist.

The C++ twin is `y3_cluster::BselSunayama23` in
`selection_boost/Bsel.hh`. It uses the same row-label contract and also
provides a DataBlock constructor.

## Dispatch and selected-profile consumers

`BselModels` in `selection_boost/Bsel.py` and `selection_boost/Bsel.hh`
dispatches only the two explicit tags:

```text
Costanzi26  -> BselCostanzi26  -> boost_selection_costanzi26
Sunayama23  -> BselSunayama23  -> bsel_profile_sunayama23
```

The profile consumers remain physically separate from the kernels:

- `Shear1h2hMaxSelCostanzi26` multiplies a local profile using
  `(R, lob, z)`;
- `Shear1h2hMaxSelSunayama23` multiplies a local profile using
  `(R, bin_index)`.

This boundary is what keeps the Bsel factor from changing the quadrature,
selection weights, or `shear1h2h_max/vals` layout. The Python consumer is
loaded by `selected_shear1h2h_max_consumer(...)`; the C++ consumer is created
by `make_shear1h2h_max_selected_profile(...)` in `Bsel.hh`.

Unknown tags, missing model sections, invalid parameter values, and absent
Sunayama bin labels fail explicitly. There is no fallback from one model to
the other.

## CosmoSIS module contract

The `Shear1h2hMax` module reads the following selection options:

| Option | Meaning | Default |
|---|---|---|
| `bsel` | Enable the tagged optical selection boost | `F` |
| `bsel_model` | Select `Costanzi26` or `Sunayama23` | required when `bsel = T` |

The model tag determines the standard values section. The module does not
read a separate `bsel_section` option. The input profile tables and
selection-weight DataBlock contract are unchanged. The quadrature options
also remain on `Shear1h2hMax`: `n_lnm` defaults to `96`, `n_z` defaults to
`64`, and `zt_low`, `zt_high`, `lnm_low`, and `lnm_high` define their ranges.
Bsel is evaluated on the resulting local profile and does not change those
quadrature grids. The module writes:

```text
shear1h2h_max/vals
```

with bin slow and radius fast ordering. The ordinary max-model result is
unchanged when `bsel = F`.

## Other systematics configurations

```ini
[sel_function]
file = ${Y3_CLUSTER_CPP_DIR}/src/pipelines/systematics/selection_richness/python/sel_function.py

[bsel]
file = ${Y3_CLUSTER_CPP_DIR}/src/pipelines/systematics/selection_bias/python/bsel.py
```

`prj_params.py` is loaded explicitly by configurations that need to publish
the projection coefficients.

## Validation

The tagged kernels and dispatchers are covered by:

- `test/bsel_profile_models.test.py` — Python values, derivatives, wall
  validation, DataBlock loading, selected consumers, and numerical precision;
- `test/bsel_profile_models.test.cc` — C++ kernels, DataBlock constructors,
  tag dispatch, and selected consumers;
- `test/shear1h2h_max.test.py` and
  `test/shear1h2h_max_t.test.cc` — Python/C++ max-profile integration and
  preservation of the output contract.
