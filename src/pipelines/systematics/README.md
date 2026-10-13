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
│   ├── Bsel.py                # tagged dispatcher and BselDeltaSigmaOperator
│   ├── BselCostanzi26.py     # Costanzi26 B_sel(R) kernel
│   ├── BselSunayama23.py     # Sunayama23 B_sel(R) kernel
│   └── Bsel.hh                # C++ kernels, dispatcher, and operator
├── miscentering/
│   ├── Miscentering.hh        # Miscentering (stack) + GL operator + Cuhre reference
│   └── Miscentering.py        # Python twin of the gauss_legendre path
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
boost applied to the max-model profile by `Shear1h2hMax` (C++ and Python).
It is enabled in the module configuration with `bsel = T` and selected
with the exact tag `Costanzi26` or `Sunayama23`.

The boost multiplies the 3D-projected *surface density* $\Sigma$, so the
excess surface density picks up a non-local term. For each
$(\ln M, z)$ node,

```text
DeltaSigma_sel(R) = B(R) DeltaSigma(R) - M_sel(R) / (pi R^2)
M_sel(R) / (pi R^2) = R int_0^1 u^2 barSigma(<uR) B'(uR) du
```

- `profile_method = piecewise_approx`: `BselDeltaSigmaOperator`
  evaluates the `u` integral with `n_gl = 20` Gauss-Legendre nodes, using
  `barSigma(<r)` assembled from the halo-model tables (`Sigma_bar_nfw`,
  `Sigma_bar_hh`, `delta_gap_0`). Sunayama23's `B'` jumps at `R0`
  (`pi0/R0 -> c/R`), so the integral is split at `u = R0/R`
  (`BselModels.break_radius`: `R0` for Sunayama23, 0 for the smooth
  Costanzi26). Against the `gauss_legendre` full integral on the real
  `des_y3_cpp0d_fast_apriori.ini` block, the piecewise path agrees to
  <= 0.41% with Costanzi26 and to 0.56% at R = 5 Mpc/h with Sunayama23
  (2.9% before the split).
- `profile_method = gauss_legendre`: the equivalent local form
  `DeltaSigma_sel = (2/R^2) int_0^R R' B Sigma dR' - B(R) Sigma(R)`,
  i.e. `B` weights the aperture nodes; no extra tables.

The `shear1h2h_max/vals` output section and bin/radius wall are the same
with and without Bsel. The Python and C++ implementations are identical
(`piecewise_approx` parity ~5e-15).

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
A = 0.03
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
Scalar `pi0`/`r0` (no `lambda_bin`) apply to every bin. The standard
DataBlock section is `[boost_selection_sunayama23]` (the legacy name
`[bsel_profile_sunayama23]` is still accepted):

```ini
[Shear1h2hMax]
bsel = T
bsel_model = Sunayama23

[boost_selection_sunayama23]
pi0 = 0.45
r0 = 3.5
c = -0.1
; or per-bin rows:
; lambda_bin = 0 1 2 3
; pi0 = 0.20 0.45 0.45 0.45
; r0 = 2.0 3.5 3.5 3.5
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

## Dispatch and the Bsel operator

`BselModels` in `selection_boost/Bsel.py` and `selection_boost/Bsel.hh`
dispatches only the two explicit tags:

```text
Costanzi26  -> BselCostanzi26  -> boost_selection_costanzi26  (or legacy bsel_profile_costanzi26)
Sunayama23  -> BselSunayama23  -> boost_selection_sunayama23  (or legacy bsel_profile_sunayama23)
```

`BselDeltaSigmaOperator` (Python `Bsel.py`, C++ `Bsel.hh`) wraps a
`BselModels` instance and applies the non-local correction above; it is
what `Shear1h2hMax` uses on the `piecewise_approx` path.

The multiply-only helpers `Shear1h2hMaxSelCostanzi26`,
`Shear1h2hMaxSelSunayama23`, `selected_shear1h2h_max_consumer(...)`
(Python) and `make_shear1h2h_max_selected_profile(...)` (C++) remain in
the library and its tests, but `Shear1h2hMax` no longer uses them:
`B DeltaSigma` alone drops the `M_sel/(pi R^2)` term.

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
read a separate `bsel_section` option. `lob_centers` (default
`25 37.5 52.5 130`) anchors the Costanzi26 richness scale. With
`profile_method = piecewise_approx`, `bsel = T` additionally requires
`haloModel/{Sigma_bar_nfw, Sigma_bar_hh, delta_gap_0}`; `gauss_legendre`
needs nothing extra. The quadrature options (`n_lnm`, `n_z`, `zt_low`,
`zt_high`, `lnm_low`, `lnm_high`, `n_los`, `n_aperture`) are unchanged by
Bsel. The full option list and a quick-run recipe are in
`../des_y3/shear_1h2h/README.md` ("How to run `Shear1h2hMax`"). The
module writes:

```text
shear1h2h_max/vals
```

with bin slow and radius fast ordering. The ordinary max-model result is
unchanged when `bsel = F`.

### Quick run with Bsel

Set the values from the command line with `-v` (after
`source ~/cosmosis_y3/cosmosis_init_macos.sh` on macOS):

```bash
cosmosis cosmosis-models/des_y3_cpp0d_fast_apriori.ini -p runtime.sampler=test test.save_dir=out \
  "pipeline.modules=consistency GrowthFactor cp_camb MfTinker halo_model \
    average_sigma_crit_inv sel_function Shear1h2hMax" \
  Shear1h2hMax.bsel=T Shear1h2hMax.bsel_model=Costanzi26 \
  -v boost_selection_costanzi26.A=0.03 boost_selection_costanzi26.alpha=0.92 \
     boost_selection_costanzi26.beta=-0.53 boost_selection_costanzi26.gamma=4.1
```

## Other systematics configurations

```ini
[sel_function]
file = ${Y3_CLUSTER_CPP_DIR}/src/pipelines/systematics/selection_richness/python/sel_function.py

[bsel]
file = ${Y3_CLUSTER_CPP_DIR}/src/pipelines/systematics/selection_bias/python/bsel.py
```

`prj_params.py` is loaded explicitly by configurations that need to publish
the projection coefficients.

## Miscentering with Gamma kernel

`miscentering/Miscentering.hh` (Python twin `Miscentering.py`) computes
the gamma-miscentred excess surface density consumed by `Shear1h2hMax`
(`miscentering = T`).  The consumer calls one class, `Miscentering`: it
reads `[miscentering]` and the `mis_*` module options, takes the
consumer's stack (weights, z nodes, Bsel model) and centred profile
$\Sigma(r\mid\ln M, z)$ each sample, and returns $O_{\rm mis}(b, R)$
and the mixture.  Inside, `mis_method` picks `MiscenteringOperator`
(fixed GL, default) or `MiscenteringCuhre` (adaptive reference).  For a
halo with centred surface density
$\Sigma(r)$ and offset scale $\sigma = \tau_{\rm mis}R_\lambda(\lambda)$,

$$
\Sigma_{\rm mis}(R) = \int_0^\infty dR_s\,\frac{R_s}{\sigma^2}e^{-R_s/\sigma}
\,\frac{1}{\pi}\int_0^\pi d\theta\;\Sigma\!\left(\sqrt{R^2+R_s^2+2RR_s\cos\theta}\right),
\qquad
\Delta\Sigma_{\rm mis}(R) = \int_0^1 4t^3\,\Sigma_{\rm mis}(Rt^2)\,dt - \Sigma_{\rm mis}(R).
$$

**Why through $\Sigma$.**  $\Delta\Sigma$ is defined relative to a centre,
so it cannot be ring-averaged like $\Sigma$: for a point mass at offset
$R_s$ the truth is $0$ inside and $M/\pi R^2$ outside, the ring average
$M/\pi|R^2-R_s^2|$ on both sides.  The map $\Delta\Sigma\to\Delta\Sigma_{\rm mis}$
is still linear — in Hankel space $\Sigma$ and $\Delta\Sigma$ share
$\tilde\Sigma(k)$ ($J_0$ / $J_2$) and the gamma kernel multiplies it by
$\tilde P(k) = (1+k^2\sigma^2)^{-3/2}$ — which the tests use as the
analytic truth; no Fourier–Bessel transform runs in the code.

**The numerical trick — one weight row per $(\sigma, R)$.**  The kernel
depends only on radius, never on $(\ln M, z)$.  Tabulate $\Sigma$ once per
sample on a log-uniform grid $r_m$ (linear in $\ln r$ between nodes); then
the fixed-GL evaluation of both integrals is linear in the tabulated
values,

$$
\Delta\Sigma_{\rm mis}(R) = \sum_m L_{\sigma,R,m}\,\Sigma(r_m),
$$

with `MiscenteringOperator::excess_surface_density_row` building $L$ from
offset nodes on two panels split at $R_s = R$ (where the ring crosses the
halo centre), angle nodes $\theta = \pi(1-v^2)$ clustered at $\theta=\pi$,
and the aperture map $R' = Rt^2$ (weights $4wt^3$, as
`OperatorMax1h2h_gl`).  Rows are built once per $(\sigma_b, R)$ and reused
while $\tau_{\rm mis}$ is unchanged; a stacked consumer folds the
selection weight in once, $g_b[m] = \sum_{k,q} W\,B\,\Sigma(r_m\mid k,q)$,
and evaluates $O_{\rm mis} = L\cdot g_b$.

**Keep the core mass.**  $\Delta\Sigma$ remembers enclosed mass through
$1/R^2$: clamping $\Sigma$ below 0.1 Mpc/h biases $\Delta\Sigma_{\rm mis}$
of an NFW by −23% at $R=0.2$ and −1% at $R=5$.  `LogRadialGrid(...,
core = true)` adds a core slot: ring nodes with $r < r_{\rm lo}$ count as a
uniform disk of the projected core mass (`haloModel/m2d_core`); with
$r_{\rm lo} = 10^{-3}$ that is as accurate as the exact NFW interior
(1.4e-5).

**Reference.**  `MiscenteringCuhre` integrates both equations as one 3-D
adaptive Cuhre integral over $(t, y, \theta)$ ($R'=Rt^2$, offsets
$x = y/(1-y)$ on $[0,\infty)$), no tables of $\Sigma_{\rm mis}$, from a
callable profile or a cluster-toolkit-style spline.  Smooth input
matters: a linearly interpolated table stalls it at a 3e-4 error
estimate, hence `CubicSplineLnR` (natural cubic in $\ln r$).

**Precision** (analytic NFW, $M_{200m}=2\times10^{14}$, $c=4$,
$\sigma=0.149$, vs the Hankel truth, $R\in[0.2,5]$): Cuhre 3e-8; GL
operator on analytic $\Sigma$ 3.4e-7 (32/32/32 nodes); GL on a table
from $10^{-3}$ plus core disk, $N_r = 1024$: 1.4e-5; cluster-toolkit
2.9e-5.  On the real `Shear1h2hMax` stack, GL vs Cuhre 2.5e-5 max.

## Validation

The tagged kernels and dispatchers are covered by:

- `test/bsel_profile_models.test.py` — Python values, derivatives, wall
  validation, DataBlock loading, selected consumers, and numerical precision;
- `test/bsel_profile_models.test.cc` — C++ kernels, DataBlock constructors,
  tag dispatch, and selected consumers;
- `test/shear1h2h_max.test.py` and
  `test/shear1h2h_max_t.test.cc` — Python/C++ max-profile integration and
  preservation of the output contract (incl. the tabulated Bsel wall path
  == per-node `apply()`);
- `test/miscentering_operator.test.cc` — miscentering operator and Cuhre
  reference vs Hankel-space NFW pins
  (`test/make_miscentering_nfw_truth_pins.py`), row vs callback form, the
  core slot, and the DataBlock contract.

