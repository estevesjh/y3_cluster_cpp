# Selection Bias

`C++` + `Python` · `y3_cluster_cpp` · `Systematics` · modules `b_sel_marg` (66 ms/sample) and `bsel` (few ms)

Optical cluster finders select on richness, and richness is boosted by
galaxies projected along the line of sight. A richness-selected sample
is therefore biased towards haloes with *excess correlated structure*
along the line of sight, and its effective clustering bias is not the
halo bias $b(M,z)$ of the underlying haloes but a **scale-dependent,
selection-affected bias** $b_{\rm sel}(\theta)$: enhanced within the
richness aperture, where the projected neighbours that inflated the
richness live, and relaxing towards the large-scale value outside it.
This page documents the two-stage pipeline that closes
$b_{\rm sel}(\theta)$ per observed richness/redshift bin from the
richness–mass model, following
[Costanzi et al. 2026, PhRvD 113, 103508](https://ui.adsabs.harvard.edu/abs/2026PhRvD.113j3508C/abstract)
(arXiv:[2604.05833](https://arxiv.org/abs/2604.05833)); the derivation
is summarised in {doc}`../math/index`.

1. **`b_sel_marg`** (C++): evaluates the population operators
   $P_1$, $I_1$, $J$ on the observed `(lambda_bin, zo_low, zo_high)`
   wall — the richness a random line of sight, and the correlated line
   of sight, adds to a cluster of true richness $\lambda^{\rm tr}$.
2. **`bsel`** (Python): closes those operators into exactly two numbers
   per wall row, the small- and large-angle plateaus
   $B_{\rm small}$ and $B_{\rm large}$.

Every downstream consumer — the projection-shear backends — reads the
same row key and the same two numbers and reconstructs the angular
dependence analytically; none interpolates between redshift rows or
consumes a $\theta$ lookup table.

## Physics

### Why the selected bias is scale dependent

Write the observed richness as $\lambda^{\rm ob} = \lambda^{\rm tr} +
\Delta^{\rm prj}$, where $\Delta^{\rm prj}$ is the richness projected
from other haloes along the line of sight within the photo-$z$ window
and inside the aperture $R_\lambda$. Conditioning on
$\lambda^{\rm ob}$ at fixed $\lambda^{\rm tr}$ conditions on
$\Delta^{\rm prj}$, hence on the line-of-sight overdensity. The excess
neighbours responsible for $\Delta^{\rm prj}$ sit at
$\theta \lesssim \theta_\lambda = R_\lambda(1+z^{\rm ob})/\chi(z^{\rm ob})$;
outside that aperture the selection only knows about the
long-wavelength environment through the halo's own bias. The
Costanzi-2026 ansatz for the halo–halo correlation of a selected
cluster with a neighbour of mass $M$ is therefore

$$\xi_{hh}(r, \theta \mid M) = b(M,z)\,b_{\rm sel}(\theta)\,\xi_{\rm NL}(r),
\qquad
b_{\rm sel}(\theta) = B_{\rm small} + (B_{\rm large} - B_{\rm small})\,\sigma(\theta),$$

with the sigmoid transition

$$\sigma(\theta) = \left[1 + \exp\left(-\frac{2.5}{\theta_\lambda}
\left(\theta - \frac{\theta_\lambda}{2}\right)\right)\right]^{-1},
\qquad
\theta_\lambda = \frac{R_\lambda(\lambda^{\rm ob})\,(1+z^{\rm ob})}{\chi(z^{\rm ob})},
\quad R_\lambda = (\lambda^{\rm ob}/100)^{0.2}.$$

The two plateaus are what the projection shear needs
({doc}`../observables/shear_projection`); their values follow from
demanding consistency between the richness the model attributes to
projection and the correlated structure the bias implies.

### The population operators $\mathcal P[X]$

For an observed bin $(\lambda^{\rm ob}, z^{\rm ob})$ the expected
projected richness contributed by neighbours weighted by $X$ is

$$\mathcal P[X](\lambda^{\rm ob}, z^{\rm ob}) =
\int d\theta\,2\pi\sin\theta \int dz\,\frac{dV}{dz\,d\Omega}\,w_z(z, z^{\rm ob})
\int d\ln M\, n(M,z) \int_0^{\lambda^{\rm ob}} d\lambda^{\rm tr}\,
\lambda^{\rm tr}\, P(\lambda^{\rm tr}\mid M, z)\;
\mathcal E(\theta)\; X(\theta, M, z),$$

with $w_z$ the photo-$z$ window, $\mathcal E(\theta)$ the aperture
membership fraction, and the $\lambda^{\rm tr}$ integral on
$(0, \lambda^{\rm ob}]$ per grid point. The three operators
`b_sel_marg` publishes are

$$
\begin{aligned}
P_1 &= \mathcal P[1] &&\text{(random line of sight: richness from uncorrelated neighbours)},\\
I_1 &= \mathcal P[b\,\xi_{\rm NL}\,\sigma(\theta)] &&\text{(correlated neighbours, large-angle weight)},\\
J &= \mathcal P[b\,\xi_{\rm NL}\,(1-\sigma(\theta))] &&\text{(correlated neighbours, small-angle weight)},
\end{aligned}
$$

so that $I_2 \equiv \mathcal P[b\,\xi_{\rm NL}] = I_1 + J$. $J$ is
evaluated directly rather than as $I_2 - I_1$, which would subtract two
nearly equal numbers. Two factors are deliberately **absent**: the
survey area $\Omega(z)$ and the Poisson normalisation $B_i$ cancel in
every downstream ratio and are not in the Python reference either
({doc}`../modules/survey_area`). In the paper's notation `P1` is
$\mathcal P[1]$, $I_2$ is $\mathcal P[b\,\xi_{\rm NL}]$ and $I_1$ is
$\mathcal P[b\,\xi_{\rm NL}\,\sigma(\theta)]$.

### The closure

For each wall row the Python stage first evaluates the
halo-mass-weighted effective bias of the haloes hosting the bin's
clusters (paper: $b_{\rm halo}$),

$$b_{\rm eff}(\lambda^{\rm ob}, z^{\rm ob}) =
\frac{\int d\ln M\,\frac{dn}{d\ln M}\,P_{\rm HOD}(\lambda^{\rm ob}\mid M, z^{\rm ob})\,M\,b(M, z^{\rm ob})}
     {\int d\ln M\,\frac{dn}{d\ln M}\,P_{\rm HOD}(\lambda^{\rm ob}\mid M, z^{\rm ob})\,M}.$$

Then, on Gauss–Legendre nodes in the true richness
$\lambda^{\rm tr} \in [\lambda_{\rm lo}, f_{\rm hi}\lambda^{\rm ob}]$,
it solves the two consistency conditions. The richness a cluster with
the *average* bias would collect from projection is
$\Delta_{\rm RND} = P_1 + b_{\rm eff}(I_1 + J)$; the large-angle
plateau is anchored to $b_{\rm eff}$ with a calibrated 13% response to
the excess projected richness,

$$b_{\rm large}(\lambda^{\rm tr}) = b_{\rm eff}\left[1 + 0.13\left(
\frac{\lambda^{\rm ob} - \lambda^{\rm tr}}{\Delta_{\rm RND}} - 1\right)\right],$$

and the small-angle plateau is whatever bias inside the aperture is
needed for the correlated neighbours to account for the rest of the
projected richness,

$$b_{\rm small}(\lambda^{\rm tr}) =
\frac{(\lambda^{\rm ob} - \lambda^{\rm tr}) - P_1 - b_{\rm large}(\lambda^{\rm tr})\,I_1}{J}.$$

The published scalars are the averages of these latent quantities over
the posterior of $\lambda^{\rm tr}$ given $\lambda^{\rm ob}$ — the
EMG projection kernel $P(\lambda^{\rm ob}\mid\lambda^{\rm tr}, z)$
({doc}`../modules/richness_mass`) times the HOD prior on
$\lambda^{\rm tr}$ marginalised over mass. A cluster whose observed
richness exceeds its true richness by more than a random line of sight
provides ($\lambda^{\rm ob} - \lambda^{\rm tr} > \Delta_{\rm RND}$) is
assigned $b_{\rm small} > b_{\rm large} > b_{\rm eff}$: more correlated
structure inside the aperture, a mildly enhanced large-scale bias. The
full angular value is reconstructed by every consumer as

```text
b_sel(theta) = b_small + (b_large - b_small) * sigmoid(theta, lob, zob)
```

No $\theta$ quadrature belongs in the Python closure.

## One wall, one row

The wall is the zipped C++ integration grid. Its row identity is

```text
(lambda_bin, zo_low, zo_high)
```

The observed-richness centre is not configured independently: it is
read from `sel_function/lambda_edges` as
`lob = 0.5 * (lambda_edges[lambda_bin] + lambda_edges[lambda_bin + 1])`,
and the observed-redshift centre is `zob = 0.5 * (zo_low + zo_high)`.
This gives an exact, aligned vector rather than a rectangular
`(zob, lob)` array; for the 12-row DES wall every output vector has
length 12 in the input wall order.

| Section/key | Shape | Meaning |
|---|---:|---|
| `b_sel_marg_P1/{lambda_bin, zo_low, zo_high, vals}` | `(N_wall,)` | C++ wall key and $P_1$ |
| `b_sel_marg_I1/vals` | `(N_wall,)` | C++ $I_1$ |
| `b_sel_marg_J/vals` | `(N_wall,)` | C++ $J = I_2 - I_1$ |
| `b_sel_marginalised/lambda_bin` | `(N_wall,)` | copied wall key |
| `b_sel_marginalised/{zo_low, zo_high, zob, lob}` | `(N_wall,)` | copied/derived wall geometry |
| `b_sel_marginalised/{b_small, b_large}` | `(N_wall,)` | one scalar pair per wall row |

The old `b_sel_marginalised/{theta, vals}` table and rectangular
`b_small/b_large` arrays are not part of the production contract. Dump
replay can still read the old rectangular format and expand it into
exact rows, but new pipeline runs never write it.

## Stage 1: `b_sel_marg` (C++)

Source model
[`src/models/p_operator_t.hh`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/models/p_operator_t.hh),
module [`src/modules/b_sel_marg_cpu/BSelMargIntegrand.cc`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/modules/b_sel_marg_cpu/BSelMargIntegrand.cc)
(`BSelMargIntegrand.so`). The three operators are co-computed in one
fixed-GL pass over $(\lambda^{\rm tr}, \ln M, \theta, z)$ — `n_lt`
nodes in true richness, `n_lnm` in mass, `n_theta` in angle, and an
exclusion-ring band plus outer wings in redshift (`n_zring`,
`n_zouter`) — 0.17 s on the 12-row wall versus 208 s for the PAGANI
reference variants (`P1/I1/I2PaganiIntegrand.so`, {doc}`../variants`).

```ini
[b_sel_marg]
file = ${Y3_CLUSTER_CPP_DIR}/release-build/src/modules/b_sel_marg_cpu/BSelMargIntegrand.so
lnm_low  = 29.9336
lnm_high = 35.6814
n_lt     = 60
n_lnm    = 24
n_theta  = 10
n_zring  = 20
n_zouter = 20
zo_low     = 0.20 0.20 0.20 0.20  0.35 0.35 0.35 0.35  0.50 0.50 0.50 0.50
zo_high    = 0.35 0.35 0.35 0.35  0.50 0.50 0.50 0.50  0.65 0.65 0.65 0.65
lambda_bin = 0 1 2 3  0 1 2 3  0 1 2 3
```

The ini section must be `b_sel_marg` (hard-coded `module_label()`). The
`lambda_bin` values refer to the edges published by `sel_function`; no
second `lob` list belongs in the `[bsel]` section. Inputs: the HMF
(`mass_function/*` via `HMF_t`), `haloModel/bias`, `xi_nl`,
`distances`, the HOD parameters (`cluster_mor/*`), and the EMG
projection coefficients (`plob_ltr_params/*` or the compiled-in
default).

## Stage 2: `bsel` (Python)

The maintained implementation is
[`src/pipelines/systematics/selection_bias/python/bsel.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/pipelines/systematics/selection_bias/python/bsel.py);
`src/pipelines/cosmology/bsel.py` is the compatibility twin (identity
pinned by `test/dual_copy_identity.test.py`), and
`src/pipelines/shared/datablock_models.py` owns the data and HOD models
shared with `sel_function`.

```ini
[bsel]
file = ${Y3_CLUSTER_CPP_DIR}/src/pipelines/systematics/selection_bias/python/bsel.py
n_ltr         = 128
ltr_lo        = 1.0
ltr_hi_factor = 3.0
```

`bsel` must run after `sel_function` and `b_sel_marg`: it needs the
published lambda edges and the C++ $P_1/I_1/J$ vectors. It no longer
reads `lob`, `zob`, `n_theta`, `theta_lo`, or `theta_hi`.

| Option | Meaning | Reference |
|---|---|---:|
| `n_ltr` | true-richness GL nodes per wall row | 128 |
| `ltr_lo` | lower true-richness integration bound | 1.0 |
| `ltr_hi_factor` | upper bound $= f\,\lambda^{\rm ob}$ when `ltr_hi` is not set | 3.0 |
| `ltr_hi` | optional fixed upper bound; `>0` overrides the multiplier | 0 (disabled) |
| `min_mass4integral` | lower mass bound for $b_{\rm eff}$ and the HOD prior | `1e13` |
| `ln_M_max_log10` | upper mass bound in $\log_{10} M$ | 15.5 |
| `n_m_beff` | mass nodes for the HOD/bias integrals | 100 |
| `verbose` | one timing line per execution | `false` |

### Implementation objects

`shared.datablock_models` provides small, explicit datavector objects:

| Dataclass | Responsibility | Important methods |
|---|---|---|
| `HODParameters` | normalized `cluster_mor` scalar parameters | `from_source` |
| `PHOD` | vectorized $P(\lambda^{\rm tr}\mid M, z)$ and its adaptive quadrature | `from_source`, `mu_sat`, `__call__`, `make_ltr_quadrature` |
| `BSelWallVector` | reads and validates $P_1/I_1/J$ plus the exact C++ wall | `from_source`, `validate` |
| `BSelOutputVector` | validates and writes the exact `bsel` output vectors | `validate`, `write_to_datablock` |
| `BSelBins` | reads production output for consumers | `from_source`, `validate`, `values` |

`BSelBins.find_exact_row(lambda_bin, zob=...)` performs an exact key
lookup — a floating-point tolerance on equality, never a neighbouring
redshift row. `IntegratorGLBSel` is the numerical engine
(`from_options`, `make_ltr_quadrature`, `evaluate_b_eff`,
`evaluate_ltr_prior`, `evaluate_b_large`, `evaluate_b_small`,
`integrate_one_wall_row`, `integrate_b_small_large`); its GL nodes
depend only on the `[bsel]` options and are built once at setup.
`PrjParams` (the EMG coefficient grid, `plob_ltr_params` or the frozen
default, {doc}`sel_function`) is loaded on the first `execute` and
cached, because the table is fixed for the run; `PHOD`, `HMF`,
`Bilinear2D` (bias), and `BSelWallVector` are rebuilt per sample since
they depend on sampled parameters. `execute` reads
`sel_function/lambda_edges`, constructs `PHOD` and `BSelWallVector`,
runs the integrator, and writes one `BSelOutputVector`.

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
| `ShearPrjGl` (reference) and the DES Y1 `ShearPrjEvaluator`/`ShearPrjGsl`/`ShearPrjCuhre` | `src/pipelines/systematics/shear_prj/cpp/sigma_prj_t.hh` (`ShearPrjCore`) | `sp_detail::BSelBins::at` |
| frozen CPU (`shear_prj_frozen_physics`, `ShearPrjFrozenCuhre`) | `sigma_prj_frozen_t.hh`, `sigma_prj_frozen_interp_t.hh` | `sp_detail::BSelBins::at` |
| exact-$z$ Python `0d` | `shear_prj_gl.py` | `dm.BSelBins.values` |
| frozen CUDA `0d` | `ShearPrjFrozenGpu.cu` | `sp_detail::BSelBins::at` |
| adaptive CUDA `3d` | `DSigmaPrj3dGpu.cu` | `sp_detail::BSelBins::at` |

The C++ helper is
[`src/pipelines/systematics/selection_bias/cpp/bsel_bins_t.hh`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/pipelines/systematics/selection_bias/cpp/bsel_bins_t.hh)
(`src/models/bsel_bins_t.hh` is the twin used by the legacy production
modules). It validates vector alignment, redshift midpoints, and
duplicate exact rows.

## Compatibility and validation

`BSelBins.from_source` retains a read-only compatibility path for old
CosmoSIS test dumps containing rectangular `lob`, `zob`, `b_small`, and
`b_large` arrays; it expands them into exact rows and does not restore
interpolation. Relevant checks:

- `test/bsel_bins.test.cc`: the exact-wall-row lookup contract;
- `test/dual_copy_identity.test.py`: `systematics` vs `cosmology`
  `bsel.py` identity;
- `test/sel_function.test.py`: shared HOD and selection-kernel behavior;
- `test/shear_prj_gl.test.cc`, `test/shear_prj_gl_wrapper.test.cc`:
  wall geometry, $\theta$ grid, and $b_{\rm sel}(\theta)$ linearity in
  the clustered channel only;
- `test/dsigma_prj_3d_gpu.test.cu`, `test/shear_prj_frozen_gpu.test.cu`:
  CUDA formula assembly and frozen dump replay.

When adding a wall-bin test, populate all seven output vectors and use
the same finite redshift interval in the test wall and the bsel
fixture: a test point with `zo_low == zo_high` is not a valid production
row because the midpoint validation requires a positive-width bin.
