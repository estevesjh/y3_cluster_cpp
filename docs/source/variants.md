# Pipeline variants

The main walkthrough ({doc}`running`) documents the DES Y3 reference
pipeline (`des_y3.ini`), which swaps three DES Y1 production modules
for algorithmically-identical `src/pipelines/des_y3` fixed-GL (formerly `fast_mass`)
implementations. This page lists: the DES Y1 pipeline those modules
replace, and every other retained variant — what changes, which
observable changes, its status (comparative / validation-only), and how
its theory vector differs from the reference.

## DES Y1 pipeline — `mock_mcmc_buzzard.ini`

The production pipeline `des_y3.ini` mirrors, byte-identical except for
the three observable stages:

| DES Y3 reference (`des_y3.ini`) | DES Y1 pipeline (`mock_mcmc_buzzard.ini`) | Relationship |
|---|---|---|
| `NumCountsSijGl` | `NumCountsSel` | algorithmically identical ("by identity") |
| `Shear1hGl` | `Shear1hMisSel` | bitwise-equivalent |
| `ShearPrjGl` | `shear_prj_frozen_physics` | same `ShearPrjCore`/`ShearPrjFrozenPhysics` family, exact-$z$ instead of frozen |

Everything else — `consistency`, `GrowthFactor`, `cp_camb`, `MfTinker`,
`halo_model`, `average_sigma_crit_inv`, `sel_function`, `b_sel_marg`,
`bsel`, `likelihoods` — is the same module in both pipelines. See
{doc}`running`'s warning about `likelihood_cp.py` section-name
compatibility before running `des_y3.ini` end-to-end.

| Variant | Modules changed | Observable changed | Status |
|---|---|---|---|
| DES Y1 pipeline (`mock_mcmc_buzzard.ini`) | `NumCountsSijGl`/`Shear1hGl`/`ShearPrjGl` → `NumCountsSel`/`Shear1hMisSel`/`shear_prj_frozen_physics` | none (same theory vector) | previous-generation reference, see above |
| widePlanck self-closure (`mock_mcmc_cp_camb.ini`) | none (data + grids + `unity = T`) | $\Delta\Sigma$ on 10 radii instead of $\gamma_t$ on 15 | comparative (closure + sampler A/B) |
| Mock data-vector writer (`generate_mock_dv.ini`) | `likelihoods` → `generate_mock_dv` | none (writes the DV) | tooling |
| Full projection evaluator (`shear_prj`) | `shear_prj_frozen_physics` → `ShearPrjEvaluator` | same $\gamma_t^{\rm prj}$, exact $z$-resolved clustered channel | validation-only |
| Adaptive projection backends (`ShearPrjGsl`, `ShearPrjCuhre`, `ShearPrjFrozenCuhre`) | projection stage | same, adaptive quadrature | validation-only |
| Centred one-halo (`Shear1hSel`) | `Shear1hMisSel` → `Shear1hSel` | $\gamma_t^{1h}$ without miscentering | historical / validation |
| PAGANI selection-bias benchmarks (`P1/I1/I2PaganiIntegrand`) | `b_sel_marg` → 3 GPU modules | same $(P_1, I_1, I_2)$ | validation-only |
| **Traditional $1h{+}2h$ max model** (`Shear1h2hMax`) | `+ halo_model` 2h branch, no projection stage | $\gamma_t = \max(\gamma^{1h}, \langle b\rangle\,\Delta\Sigma_{2h})\,\Sigma_{\rm crit}^{-1}$ | **model option** (see below) |
| Population diagnostics (`MassWeightedSel`, `BiasWeightedSel`, `n_operator_ratios`) | added after `NumCountsSel` | adds $\langle M\rangle_i$, $\langle b\rangle_i$ | diagnostics |

## widePlanck self-closure — `mock_mcmc_cp_camb.ini`

The **module list is byte-identical** to the Buzzard reference; only data
and tuning differ:

- **Data vector**: `mock_dv_widePlanck_jkcov.npz` — the pipeline's own
  fiducial prediction (12 NC + **120** shear points, 10 radii
  $R \in [0.2, 5]$ cMpc/$h$), with diagonal Poisson number-count
  covariance and the Buzzard jackknife shear covariance. The test:
  $\log L \approx 0$ at the fiducial point and posterior recovery of the
  10 varied parameters. (The Buzzard run instead fits an external
  simulation measurement — recovery bias is the signal.)
- **Observable convention**: `average_sigma_crit_inv` runs with
  `unity = T`, so the "shear" vector is $\Delta\Sigma$ rather than
  $\gamma_t$.
- **Emulator files**: the `_s8_` artifact variants
  (`camb_linear_s8_v3c_emulator.npz`, `camb_linear_nonu_s8_v3c_…`).
- **Tuning**: `Shear1hMisSel.eps_rel = 5e-3` (legacy knob),
  `halo_model` `R_perp` grid 0.1–20, `[output] lock = F`, and a correct
  CosmoSIS `[polychord]` block (`live_points = 500`,
  `num_repeats = 60`, `tolerance`, `base_dir`, …) — the Buzzard ini's
  `[polychord]` keys are stale non-CosmoSIS names and must be overridden
  on the command line ({doc}`running`).
- This ini hosted the linear-vs-log-space likelihood A/B (PolyChord jobs
  55014977 / 55040404): statistically identical posteriors and identical
  convergence cost, so `log_space = F` stays the default.

`generate_mock_dv.ini` reuses the same 12 forward modules with the
likelihood replaced by the `generate_mock_dv` finisher, which writes the
self-closure `.npz`.

## Projection-stage backends

All share the physics of {doc}`observables/shear_projection`; each
hard-codes its own `module_label()`, so ini section names must match.
They also all write their outputs under distinct sections **except** that
`ShearPrjFrozenPhysics` aliases `shear_prj/*` — never co-load it with
`ShearPrjEvaluator`.

- **`ShearPrjEvaluator`** (section `shear_prj`,
  `src/models/sigma_prj_t.hh` over `sp_detail::ShearPrjCore`) — the
  full-fidelity fixed-GL evaluator: the clustered channel keeps
  $n(M,z)\,b(M,z)$ resolved in $z$ instead of frozen at $z^{\rm ob}$,
  and $\Omega(z)$ is hard-excluded. Outputs `shear_prj/{vals, rnd, cl}`.
  Sibling wrappers over the same core publish `sigma_prj/*`
  ($\Sigma^{\rm prj}$) and `dsigma_prj/*` ($\Delta\Sigma^{\rm prj}$) for
  diagnostics. The frozen module agrees to $< 0.2\%$ at the reference
  settings and is $\sim 3.2\times$ faster.
- **`ShearPrjGsl`** (section `shear_prj_gsl`, defined but not loaded in
  the Buzzard ini) — GSL QAGP outer-$z$ adaptive integration with an
  explicit breakpoint at $z = z^{\rm ob}$ for the
  $\xi_{\rm NL}(|\Delta\chi|)$ cusp; includes $\Omega(z)$.
- **`ShearPrjCuhre`** — adaptive Cuhre on the inner $(z, \ln M)$;
  $\sim 30\times$ slower; head-to-head convergence checks.
- **`ShearPrjFrozenCuhre`** ("Option E",
  `src/models/sigma_prj_frozen_interp_t.hh`) — the frozen-physics
  reduction driven by continuous Cuhre instead of the fixed grid.

## Centred one-halo — `Shear1hSel`

`NOperatorSelRadial<Shear1hWeight>` (Cuhre-driven,
`src/modules/num_counts_sel/Shear1h.cc`), writing `shear1hsel/vals`: the
$f_{\rm mis} = 0$ limit of `Shear1hMisSel`. Setting
`miscentering/f_mis = 0` in the reference module reproduces it — the
closure check used when the miscentering branch landed. Earlier mock data
vectors were generated against this branch.

## PAGANI selection-bias benchmarks

`P1PaganiIntegrand` / `I1PaganiIntegrand` / `I2PaganiIntegrand`
(`src/modules/b_sel_marg_cpu/`) run one adaptive GPU integral per
(bin, operator) over the same integrand as `b_sel_marg`. Reference-only:
the fixed-GL co-computing path is $\sim 10^3\times$ faster on the wall
grid (0.17 s vs 208 s), and PAGANI computes $I_2$ rather than $J$, so it
inherits the $I_2 - I_1$ cancellation the production module avoids.

## Traditional 1h+2h max model — `Shear1h2hMax`

The pre-projection shear model: run `halo_model` with
`compute_lensing_2h = T` ({doc}`observables/second_halo_term`), and
compose the one-halo and biased two-halo terms by the **pointwise max**
(Hayashi & White 2008, the DES Y1 lensing-analysis prescription — not a
sum):

$$\Delta\Sigma_{\max}(R, \ln M, z \mid i) = \max\!\big(
\Delta\Sigma_{\rm cl}(R, \ln M \mid i),\;
b(\ln M, z)\,\Delta\Sigma_{\rm hh}(R, z)\big), \qquad
\gamma_t^{\rm theory}(R \mid i) =
N_i[\Delta\Sigma_{\max}](R)\,\langle\Sigma_{\rm crit}^{-1}\rangle / N_i[1].$$

Implemented by
[`Shear1h2hMax`](https://github.com/estevesjh/y3_cluster_cpp/blob/pipelines/des_y3/src/pipelines/des_y3/shear_1h2h/python/0d/shear1h2h_max.py)
(`shear1h2h_max.py`, C++ `Shear1h2hMax.cc`, CUDA
`Shear1h2hMaxGpu.cu`) — a **model option**, not part of the reference
pipeline ({doc}`../running`, {doc}`observables/second_halo_term`).
Unlike the one-halo operator, the two-halo term is $z$-dependent, so the
redshift integral cannot be contracted past the profile the way
the one-halo $z$-contracted `0d` path does elsewhere — `Shear1h2hMax` (also `0d`: fixed-GL only) keeps a $z$-resolved
tabulated weight and does a double fixed-GL contraction instead.

vs the reference's $1h^{\rm mis} + {\rm prj}$: no miscentering
suppression at small $R$ and no $b_{\rm sel}$ boost at large $R$
(quantified in {doc}`observables/second_halo_term`, once its comparison
figure is regenerated with this correct composition — see that page's
note). Status: **model option** — and the wired production
`SigmaTotSel`/`DSigmaTotSel` modules remain broken
({doc}`modules/historical`), which is why `Shear1h2hMax` reads
`haloModel/dSigma_hh` directly rather than through them.

Likelihood wiring: `y3_buzzard/likelihood_cp.py` consumes the max model
with `shear_max_section = shear1h2h_max` (theory =
`shear1h2h_max/vals` / $N_i$, no projection term). Setting
`is_b_proj_costanzi26 = T` multiplies that theory by the Costanzi-2026
$\mathcal{B}_{\rm prj}(R)$ selection-bias correction
(`src/pipelines/systematics/costanzi_bprj/`, App. C of arXiv:2604.05833),
with its parameters read from the values-file section `[costanzi_bprj]`
and the per-bin $(R, \lambda, z)$ from `shear_r_perp`,
`shear_lob_centers`, `shear_zbin_reps`.

## Population diagnostics

`MassWeightedSel` ($f = M$) and `BiasWeightedSel` ($f = b(M,z)$) are
Cuhre-driven siblings of `NumCountsSel` producing $N_i[M]$ and $N_i[b]$;
the unregistered `n_operator_ratios` finisher turns them into
$\langle M\rangle_i$, $\langle b\rangle_i$. Not part of the reference
likelihood.

## Retired

Reduced shear $g_t = \gamma_t/(1-\kappa)$ (denominator retired
2026-05-11 — required for the linear 1h + prj sum), the standalone
`prj_params` DataBlock module, the `red_shear_prj` module name, and the
per-(bin, $R$) Cuhre `NOperatorSel*` evaluators. Inventory:
{doc}`modules/historical`.
