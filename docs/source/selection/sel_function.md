# Selection Function

`Python` · `y3_cluster_cpp` · `Selection` · module `sel_function` · `197 ms/sample`

Pre-tabulates, once per sample, the joint richness + photo-$z$ selection
function $S_{ij}(\ln M, z)$ for all 12 observed bins on one shared grid.
`NumCountsSel` and `Shear1hMisSel` slice their bin's plane from the packed
tensor and interpolate it inside their population integrals.

## Script

- Source: [`src/pipelines/shared/sel_function.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/main/src/pipelines/shared/sel_function.py).
- Loaded by CosmoSIS as a Python module.

`src/modules/sel_function/sel_function.py` remains a thin compatibility shim
that imports this shared implementation. The CosmoSIS configurations load
the shared file directly, and `shared/sel_kernels.py` uses the same module.
The HOD and lambda quadrature are also shared with `cosmology/bsel.py` via
`shared/datablock_models.py`.

## Numerical framework

The full integral: the **richness selection function** — the probability
that a halo of mass $M$ at true redshift $z^{\rm tr}$ is observed inside
the $(i, j)$ bin — is the redMaPPer selection function
$\mathcal S_{ij}$ integrated against the intrinsic richness–mass
relation,

$$S_{ij}(M, z^{\rm tr}) =
\int_{\Delta\lambda_i}\! d\lambda^{\rm ob}
\int_{\Delta z_j}\! dz^{\rm ob}
\int_0^\infty\! d\lambda^{\rm tr}\;
P(\lambda^{\rm ob}\mid\lambda^{\rm tr}, z^{\rm tr})\,
P(z^{\rm ob}\mid z^{\rm tr}, \Delta\lambda_i)\,
P(\lambda^{\rm tr}\mid M, z^{\rm tr}).$$

Both bin integrals are analytic, so the tabulated tensor is

$$S_{ij}(\ln M, z) = S_i(\ln M, z)\,\mathcal S_j(z)
= \Big[\textstyle\sum_k W_k\,
\mathcal S_i(\lambda_k, z)\, P_{\rm HOD}(\lambda_k \mid M, z)\Big]\,
\mathcal S_j(z),$$

with the **observed-richness kernel** $\mathcal S_i$ (the closed-form
Gaussian + EMG CDF difference at the bin edges,
{doc}`../modules/richness_mass`), the **observed-redshift kernel**
$\mathcal S_j$ (a Gaussian CDF difference of width `sigma_z`,
{doc}`../modules/redshift_kernel`), and the shifted-Poisson
$P_{\rm HOD}$. Only the $\lambda^{\rm tr}$ integral is numerical.

The recipe, per sample:

1. Per $(\ln M, z)$ cell, place $N_q = 32$ Gauss–Legendre nodes in
   $\lambda^{\rm tr}$ on the adaptive bracket
   $[\mu_{\rm eff} - L\sigma_{\rm eff},\, \mu_{\rm eff} +
   L\sigma_{\rm eff}]$, with
   $\sigma_{\rm eff} = \sqrt{\mu_{\rm sat} +
   (\sigma_\lambda\mu_{\rm sat})^2}$ and $L = 6$; evaluate
   $P_{\rm HOD}$ on the full $(192, 64, 32)$ tensor with the continuous
   shifted-Poisson log-Gamma equation (narrow-Gaussian fallback where
   $\mu_{\rm sat} \le 10^{-8}$).
2. Evaluate the 8 EMG coefficient splines on the 1-D $z$ grid only
   (saves $\sim 130$ ms/sample), broadcast to
   $(\mu, \sigma, \tau, f^{\rm prj})$ at each node.
3. Compute the EMG CDF via `erfcx` at the **5 unique bin edges**
   $\{20, 30, 45, 60, 200\}$ and difference, giving all four
   $\mathcal S_i$ tables at once.
4. Contract the existing `PHOD * PLOB_LTR` integrand over true richness
   into one $(\ln M,z)$ surface per unique edge, subtract the lower-edge
   surface from the upper-edge surface for every configured richness bin,
   multiply by $\mathcal S_j(z)$, and pack into `S_stack`.

The Python kernels match the C++ models
(`src/models/mor_hod_t.hh`, `src/models/richness_kernel_t.hh`)
line-for-line. Full derivation: {doc}`../math/index`.

### Fast execution path

The refactor does not replace the production contraction with a generic
per-node Python implementation. `execute()` constructs one shared `PHOD`
datavector for the current sample and passes that object to the existing
fused `@njit` boundary. Only the normalized scalar fields cross into
`_compute_lam_nodes_and_P_HOD_nb`, because Numba cannot consume the Python
dataclass. That kernel computes the adaptive bracket, reuses the cell-level
HOD quantities across all true-richness nodes, and fills the preallocated
output arrays in place. The shared `PHOD` NumPy path is the
readable/reference implementation and is used by `bsel`; the tests compare
the fast selection path against it.

### Shared ownership and lifecycle

`setup()` owns only values that cannot change with a sampled datablock:
the configured bin edges, the common `lnM` and `z` grids, the canonical
Gauss–Legendre nodes, and the per-module projection-spline cache.
`execute()` reads the current datablock through `DataBlockSource`, constructs
the current `PHOD`, and runs the fast numerical contraction.

The compatibility helpers `_read_mor`, `_p_hod_scalar`, and `_mu_sat` remain
available to the full-ltmz reference code and unit tests, but they delegate to
`datablock_models.HODParameters` and `PHOD`; they do not carry a second HOD
implementation. The only intentionally separate HOD implementation is the
fused Numba kernel, retained for the production performance path.

The selection contraction is represented as a small set of 2-D datavectors
rather than a new observed-richness quadrature grid:

```text
weighted_hod              (lnM, z, ltr)
edge_integrals            (lambda_edge, lnM, z)
richness_selection        (bin, lnM, z)
S_stack                   (bin, z, lnM)
```

`edge_integrals` is the integrated `PHOD * PLOB_LTR` quantity. Since the
observed-richness integral is analytic, `PLOB_LTR` enters through its CDF at
the two bin edges. This avoids materializing a redundant
`(lambda_true, lambda_observed)` tensor.

## CosmoSIS setup

```ini
[sel_function]
file = ${Y3_CLUSTER_CPP_DIR}/src/pipelines/shared/sel_function.py
lam_min = 20.0  30.0  45.0  60.0   20.0  30.0  45.0  60.0   20.0  30.0  45.0  60.0
lam_max = 30.0  45.0  60.0  200.0  30.0  45.0  60.0  200.0  30.0  45.0  60.0  200.0
zob_min = 0.20  0.20  0.20  0.20   0.35  0.35  0.35  0.35   0.50  0.50  0.50  0.50
zob_max = 0.35  0.35  0.35  0.35   0.50  0.50  0.50  0.50   0.65  0.65  0.65  0.65
sigma_z = 0.03  0.03  0.03  0.03   0.03  0.03  0.03  0.03   0.03  0.03  0.03  0.03
zt_low   = 0.05
zt_high  = 0.80
lnm_low  = 29.9336
lnm_high = 36.8414
n_lnm = 192
n_z   = 20
n_z_shared = 64
L_z    = 6.0
L_lam  = 6.0
N_q    = 32
```

- Requires `Y3_CLUSTER_CPP_DIR`.
- Ordering: after `consistency` (MOR parameters live in the sampled
  sections); before `NumCountsSel` and `Shear1hMisSel`.
- The 12 array entries define the bin wall: 4 richness bins
  $\{[20,30), [30,45), [45,60), [60,200)\}$ × 3 photo-$z$ bins
  $\{[0.20,0.35), [0.35,0.50), [0.50,0.65)\}$, richness index fastest.

## Configuration options

| Option | Meaning | Units | Reference value |
|---|---|---|---|
| `lam_min`, `lam_max` | per-bin observed-richness edges | — | 12-entry wall |
| `zob_min`, `zob_max` | per-bin observed-redshift edges | — | 12-entry wall |
| `sigma_z` | photo-$z$ scatter in the redshift kernel $\mathcal S_j$ | — | 0.03 (all bins) |
| `zt_low`, `zt_high` | shared true-$z$ grid envelope | — | 0.05, 0.80 |
| `lnm_low`, `lnm_high` | shared $\ln M$ grid envelope | $\ln(M_\odot/h)$ | 29.9336, 36.8414 |
| `n_lnm` | $\ln M$ nodes — whole-pipeline optimum 192; **64 is pathological** (GL resonance, 4.5% drift on counts) | — | 192 |
| `n_z_shared` | shared $z$ nodes | — | 64 |
| `N_q` | Gauss–Legendre nodes of the $\lambda^{\rm tr}$ quadrature | — | 32 |
| `L_lam`, `L_z` | quadrature bracket half-widths | units of $\sigma$ | 6.0, 6.0 |

### Numerical precision: choosing `n_lnm`

`n_lnm = 192` is a **whole-pipeline** optimum, not a per-module one:
coarsening the $S_{ij}$ mass grid forces the downstream Cuhre-based
`NumCountsSel` to refine harder, which cancels the time this module
saves. Measured sweep (2026-05-07, wall-clock per sample):

| `n_lnm` | `sel_function` | `NumCountsSel` | Total | Note |
|---:|---:|---:|---:|---|
| 256 | 0.38 s | 0.08 s | 1.36 s | accuracy ceiling |
| **192** | 0.22 s | 0.12 s | **1.15 s** | sweet spot (reference value) |
| 128 | 0.15 s | 0.27 s | 1.22 s | Cuhre refinement eats the savings |
| 64 | — | — | — | **pathological**: GL resonance, 4.5% drift on `NumCountsSel` — avoid |

Accuracy at `n_lnm = 192` vs `256`: $<0.05\%$ on `NumCountsSel` and
`Shear1hMisSel`.

### Refactor regression and timing test

`test/sel_function_regression.test.py` runs the current module and the exact
pre-refactor implementation from Git baseline `80b2fcf` on the same in-memory
datablock. It checks that `sel_function/S_stack` agrees to `2\times10^{-6}`
relative tolerance and that the warmed-up median per-sample runtime is no more
than 50% slower. The test uses a smaller production-shaped grid so it is
appropriate for CI; the test's timing is a guard against an accidental
regression, not a replacement for the production-grid benchmark above.

Run it directly with:

```bash
python test/sel_function_regression.test.py
```

The test requires the normal CosmoSIS Python environment and a checkout that
contains the baseline Git commit.

## DataBlock inputs

| DataBlock input | Meaning | Units / shape | Produced by |
|---|---|---|---|
| `cluster_mor/{log10_Mmin, log10_M1 \| log10_ratio, alpha, epsilon, sigma_lambda}` | shifted-Poisson HOD richness–mass parameters | — | sampler (values file) |
| `plob_ltr_params/*` | EMG projection-kernel coefficient splines (optional; falls back to `PrjParams.default()`) | — | `prj_params` module / in-code fallback |

## DataBlock outputs

| DataBlock output | Meaning | Units / shape | Consumed by |
|---|---|---|---|
| `sel_function/lnM` | shared mass grid | `(192,)` | `NumCountsSel`, `Shear1hMisSel` |
| `sel_function/z` | shared redshift grid | `(64,)` | same |
| `sel_function/S_stack` | packed selection tensor $S_{ij}(\ln M, z)$, layout `(bin, z, lnM)`, C-contiguous | `(12, 64, 192)` | same |
| `sel_function/lambda_edges` | unique observed-richness bin edges used by the selection wall | `(5,)` | `bsel.py` |
| `sel_function/lambda_centres` | arithmetic centres of those edges | `(4,)` | `bsel.py` and shear consumers |
