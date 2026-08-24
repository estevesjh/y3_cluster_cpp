# Selection bias and the exact wall contract

The selection-bias pipeline has two stages:

1. the C++ `b_sel_marg` module evaluates the population operators
   `P1`, `I1`, and `J` on the observed `(lambda_bin, zo_low, zo_high)` wall;
2. the Python `bsel` module closes those operators into exactly two numbers
   per wall row: `b_small` and `b_large`.

The downstream shear implementations all use the same row key and the same
two numbers. They evaluate the angular dependence analytically; none of them
interpolates between redshift rows or consumes a theta lookup table.

The model is [Costanzi et al. 2026, PhRvD 113,
103508](https://ui.adsabs.harvard.edu/abs/2026PhRvD.113j3508C/abstract), with
the derivation summarized in {doc}`../math/index`.

## One wall, one row

The wall is the zipped C++ integration grid. Its row identity is:

```text
(lambda_bin, zo_low, zo_high)
```

The observed-richness centre is not configured independently. It is read from
`sel_function/lambda_edges` and calculated as

```text
lob = 0.5 * (lambda_edges[lambda_bin]
             + lambda_edges[lambda_bin + 1])
```

The observed-redshift centre is calculated as

```text
zob = 0.5 * (zo_low + zo_high)
```

This gives an exact, aligned vector rather than a rectangular `(zob, lob)`
array. For a 12-row DES wall, every output vector has length 12 and the row
ordering is the input C++ wall ordering.

The contract is deliberately explicit:

| Section/key | Shape | Meaning |
|---|---:|---|
| `b_sel_marg_P1/{lambda_bin, zo_low, zo_high, vals}` | `(N_wall,)` | C++ wall key and `P1` |
| `b_sel_marg_I1/vals` | `(N_wall,)` | C++ `I1` |
| `b_sel_marg_J/vals` | `(N_wall,)` | C++ `J = I2 - I1` |
| `b_sel_marginalised/lambda_bin` | `(N_wall,)` | copied wall key |
| `b_sel_marginalised/{zo_low, zo_high, zob, lob}` | `(N_wall,)` | copied/derived wall geometry |
| `b_sel_marginalised/{b_small, b_large}` | `(N_wall,)` | one scalar pair per wall row |

The old `b_sel_marginalised/{theta, vals}` table and rectangular
`b_small/b_large` arrays are not part of the production contract. Python dump
replay can still read the old rectangular format and expand it into exact
rows, but new pipeline runs never write it.

## Shared implementation

The maintained implementation lives in
[`src/pipelines/systematics/selection_bias/python/bsel.py`](/Users/jesteves/Documents/Dev/github/y3_cluster_cpp/src/pipelines/systematics/selection_bias/python/bsel.py).
`src/pipelines/cosmology/bsel.py` remains a compatibility reference, and
`src/pipelines/shared/datablock_models.py` owns the data and HOD models shared
by `sel_function` and `bsel`.

### Datavector models

`shared.datablock_models` provides the following small, explicit objects:

| Dataclass | Responsibility | Important methods |
|---|---|---|
| `HODParameters` | normalized `cluster_mor` scalar parameters | `from_source` |
| `PHOD` | vectorized `P(lambda_true | M, z)` and its adaptive quadrature | `from_source`, `mu_sat`, `__call__`, `make_ltr_quadrature` |
| `BSelWallVector` | reads and validates `P1/I1/J` plus the exact C++ wall | `from_source`, `validate` |
| `BSelOutputVector` | validates and writes the exact `bsel` output vectors | `validate`, `write_to_datablock` |
| `BSelBins` | reads production output for consumers | `from_source`, `validate`, `values` |

The important design choice is that `BSelBins.find_exact_row(lambda_bin, zob=...)`
performs an exact key lookup. It may use a numerical tolerance for floating
point equality, but it never chooses a neighboring redshift row.

### `IntegratorGLBSel`

`IntegratorGLBSel` is the numerical engine in `cosmology/bsel.py`. Its public
methods are intentionally specific:

```python
IntegratorGLBSel.from_options(options)
IntegratorGLBSel.make_ltr_quadrature(lob)
IntegratorGLBSel.evaluate_b_large(lob, ltr, p1, i1, j, b_eff)
IntegratorGLBSel.evaluate_b_small(lob, ltr, p1, i1, j, b_large)
IntegratorGLBSel.integrate_one_wall_row(
    lob, zob, p1, i1, j, phod, hmf, halo_bias, projection_parameters
)
IntegratorGLBSel.evaluate_b_eff(
    lob, zob, phod, mass, number_density, halo_bias, mass_weights
)
IntegratorGLBSel.evaluate_ltr_prior(
    ltr, zob, lob, phod, mass, number_density, mass_weights,
    projection_parameters
)
IntegratorGLBSel.integrate_b_small_large(
    wall, phod, hmf, halo_bias, projection_parameters
)
```

`projection_parameters` is a `PrjParams` dataclass. It contains the shared
redshift grid and the eight EMG coefficient arrays, and provides
`interp_linear(name, z)` for coefficient lookup and
`p_lob_given_ltr(lob, ltr, z)` / `cdf_lob(lob, ltr, z)` for the analytical EMG
density and CDF. `PHOD` and
`PrjParams` are therefore passed as model objects; neither is reconstructed
as a string-keyed dictionary inside `bsel.py`. `PrjParams` uses the complete
`plob_ltr_params` datablock table when available and otherwise selects its
complete frozen default through `from_source_or_default(source)`.

`IntegratorGLBSel.__post_init__` obtains the mass nodes and weights from
`shared.datablock_models.gl_nodes` and caches them with the canonical
true-richness Legendre nodes. These rules depend only on the `[bsel]` options,
so they are setup state and are reused for every sample and wall row.

`execute` still creates the datablock-backed `PHOD`, `HMF`, `Bilinear2D`, and
`BSelWallVector` for each sample. Their values can depend on sampled HOD,
cosmology, nuisance parameters, or the C++ wall operators, so moving them to
setup would reuse stale sample data. The projection calibration is different:
`PrjParams` is loaded once on the first execute call and then cached because
the published `plob_ltr_params` table is fixed for the run. Standard CosmoSIS
setup does not receive a datablock, which is why this one-time discovery occurs
in `get_projection_parameters` rather than in `setup`.

The module entry points are thin orchestration functions:

```python
config = setup(options)
execute(block, config)
cleanup(config)
```

`execute` reads `sel_function/lambda_edges`, constructs `PHOD` and
`BSelWallVector`, runs the integrator, then writes one `BSelOutputVector`.

## C++ population operators

`C++` · module `b_sel_marg` · source model
[`src/models/p_operator_t.hh`](/Users/jesteves/Documents/Dev/github/y3_cluster_cpp/src/models/p_operator_t.hh)

The operators are

$$
\begin{aligned}
P_1 &= \mathcal P[1],\\
I_1 &= \mathcal P[b\,\xi_{\rm NL}\,\sigma(\theta)],\\
J &= \mathcal P[b\,\xi_{\rm NL}\,(1-\sigma(\theta))].
\end{aligned}
$$

Here `J` is evaluated directly rather than formed by subtracting two nearly
equal numbers. The shared sigmoid is

$$
\sigma(\theta) = \left[1 + \exp\left(-\frac{2.5}{\theta_\lambda}
\left(\theta - \frac{\theta_\lambda}{2}\right)\right)\right]^{-1},
$$

with

$$
\theta_\lambda = \frac{R_\lambda(lob)(1+zob)}{\chi(zob)},
\qquad R_\lambda(lob) = (lob/100)^{0.2}.
$$

The C++ wall configuration remains the source of truth for row identity:

```ini
[b_sel_marg]
zo_low     = 0.20 0.20 0.20 0.20  0.35 0.35 0.35 0.35  0.50 0.50 0.50 0.50
zo_high    = 0.35 0.35 0.35 0.35  0.50 0.50 0.50 0.50  0.65 0.65 0.65 0.65
lambda_bin = 0 1 2 3  0 1 2 3  0 1 2 3
```

The `lambda_bin` values refer to the edges published by `sel_function`; no
second `lob` list belongs in the `[bsel]` section.

## Python closure

For each exact wall row, the closure first evaluates the halo-mass-weighted
effective bias

$$
b_{\rm eff}(lob,zob) =
\frac{\int d\ln M\,(dn/dM)\,P_{\rm HOD}(lob|M,zob)\,M\,b(M,zob)}
     {\int d\ln M\,(dn/dM)\,P_{\rm HOD}(lob|M,zob)\,M}.
$$

For each true richness GL node, it then evaluates

$$
\Delta_{\rm RND} = P_1 + b_{\rm eff}(I_1+J),
$$

$$
b_{\rm large}(\lambda^{\rm tr})
 = b_{\rm eff}\left[1+0.13\left(
\frac{lob-\lambda^{\rm tr}}{\Delta_{\rm RND}}-1\right)\right],
$$

$$
b_{\rm small}(\lambda^{\rm tr})
 = \frac{(lob-\lambda^{\rm tr})-P_1
       -b_{\rm large}(\lambda^{\rm tr})I_1}{J}.
$$

The two scalar outputs are the EMG/HOD-weighted averages of those latent
quantities. The full angular value is reconstructed by every consumer as

```text
b_sel(theta) = b_small + (b_large - b_small) * sigmoid(theta, lob, zob)
```

No theta quadrature belongs in the Python closure.

## CosmoSIS configuration

```ini
[bsel]
file = ${Y3_CLUSTER_CPP_DIR}/src/pipelines/systematics/selection_bias/python/bsel.py
n_ltr         = 128
ltr_lo       = 1.0
ltr_hi_factor = 3.0
```

`bsel` must run after `sel_function` and `b_sel_marg`, because it needs both
the published lambda edges and the C++ `P1/I1/J` vectors. It no longer reads
`lob`, `zob`, `n_theta`, `theta_lo`, or `theta_hi` options.

| Option | Meaning | Reference |
|---|---|---:|
| `n_ltr` | true richness GL nodes per wall row | 128 |
| `ltr_lo` | lower true richness integration bound | 1.0 |
| `ltr_hi_factor` | upper bound multiplier when `ltr_hi` is not set | 3.0 |
| `ltr_hi` | optional fixed upper bound; `>0` overrides the multiplier | 0 (disabled) |
| `min_mass4integral` | lower mass bound for `b_eff` and the HOD prior | `1e13` |
| `ln_M_max_log10` | upper mass bound in `log10(M)` | 15.5 |
| `n_m_beff` | mass nodes for the HOD/bias integrals | 100 |
| `verbose` | emit one timing line per execution | `false` |

## Downstream consumers

All consumers use the same sequence:

```text
wall row -> exact (lambda_bin, zob) lookup
          -> read lob, b_small, b_large
          -> evaluate the shared sigmoid at each theta
          -> continue with the backend-specific theta/z/mass integration
```

| Consumer | Implementation | Exact lookup |
|---|---|---|
| production CPU | `src/pipelines/systematics/shear_prj/cpp/sigma_prj_t.hh` (`ShearPrjCore`, GSL, Cuhre) | `sp_detail::BSelBins::at` |
| frozen CPU | `src/pipelines/systematics/shear_prj/cpp/sigma_prj_frozen_t.hh` and `sigma_prj_frozen_interp_t.hh` | `sp_detail::BSelBins::at` |
| fast-mass Python | `shear_prj_fast_mass.py` | `dm.BSelBins.values` |
| frozen fast-mass CUDA | `ShearPrjFrozenGpu.cu` | `sp_detail::BSelBins::at` |
| full-ltmz CUDA | `DSigmaPrjFullLtmzGpu.cu` | `sp_detail::BSelBins::at` |

The maintained C++ helper is in
[`src/pipelines/systematics/selection_bias/cpp/bsel_bins_t.hh`](/Users/jesteves/Documents/Dev/github/y3_cluster_cpp/src/pipelines/systematics/selection_bias/cpp/bsel_bins_t.hh).
The original `src/models/bsel_bins_t.hh` remains available for legacy
production modules.
It validates vector alignment, redshift midpoints, and duplicate exact rows.

## Compatibility and validation

`BSelBins.from_source` retains a read-only compatibility path for old
CosmoSIS test dumps containing rectangular `lob`, `zob`, `b_small`, and
`b_large` arrays. That path expands the saved values into exact rows; it does
not restore interpolation. New production outputs always use the exact-vector
contract above.

The relevant checks are:

- `test/sel_function.test.py`: shared HOD and selection-kernel behavior;
- `test/shear_prj_fast_mass.test.cc`: wall geometry and theta-grid behavior;
- `test/dsigma_prj_full_ltmz_gpu.test.cu`: full-ltmz CUDA formula assembly;
- `test/shear_prj_frozen_gpu.test.cu`: frozen CUDA dump replay;
- `test/shear_prj_cross_backend.test.py`: optional cross-backend comparison.

When adding a wall-bin test, populate all seven output vectors and use the
same finite redshift interval in the test wall and the bsel fixture. A test
point with `zo_low == zo_high` is not a valid production bsel row because the
row midpoint validation requires a positive-width redshift bin.
