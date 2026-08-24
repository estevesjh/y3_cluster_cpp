# Second Halo Term

`Python` (producer) · `y3_cluster_cpp` (`y3_buzzard/`) · `Lensing`

The conventional two-halo term — correlated matter around the cluster,
scaled by its plain halo bias — is, in the paper's language, the
**unselected-bias limit** of the selection-affected two-halo term
$\Sigma^{\rm prj}$ used by the reference pipeline
({doc}`shear_projection`): replace $b(M,z)\,b_{\rm sel}(\theta)$ by the
halo-bias aggregate $b_{\rm halo}$ and drop the selection modulation.
Its ingredients are computed by the
{doc}`halo_model <../cosmology/halo_model>` module, gated behind
`compute_lensing_2h`.

```{admonition} Not active in the reference pipeline
:class: important
Both the DES Y1 pipeline (`mock_mcmc_buzzard.ini`) and the DES Y3
reference (`des_y3.ini`, {doc}`../running`) run `halo_model` with
`compute_lensing_2h = F`: the reference shear composition is one-halo +
projection ({doc}`shear_projection`), which never reads the two-halo
tables. Skipping the branch saves 200–300 ms per sample. The
conventional $1h{+}2h$ **max-model composition is available as a
documented model option**, `Shear1h2hMax` — see {doc}`../variants`.
```

## Script

- Producer: [`y3_buzzard/halo_model_cosmosis.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/d7feb7504ed5dfcad84f99a1791af8a55c858aa0/y3_buzzard/halo_model_cosmosis.py)
  with the Hankel-transform backend in `y3_buzzard/haloModel.py`
  (class `ct_2hTerm`, driving the `cluster_toolkit`
  $P \to \xi \to \Sigma \to \Delta\Sigma$ chain).

## Numerical framework

The full integral: around a halo of mass $M$, the two-halo surface
density is the line-of-sight projection of the halo–matter correlation,

$$\Sigma_{2h}(R; M, z) = \bar\rho_m \int d\chi_\parallel\;
\xi_{\rm hm}\!\big(\sqrt{R^2 + \chi_\parallel^2},\, z \,\big|\, M\big),
\qquad
\xi_{\rm hm} = b(M, z)\,\xi_{\rm NL},$$

$$\Delta\Sigma_{2h}(R) = \bar\Sigma_{2h}(<R) - \Sigma_{2h}(R),
\qquad \bar\rho_m = \Omega_m\,\rho_{\rm crit},$$

and the **reference combination with the one-halo term is the
$\Sigma_{\max}/\kappa_{\max}$ ("max") model**: not the sum but the
pointwise maximum of the one-halo and biased two-halo surface
densities — the Hayashi & White (2008) prescription of the DES Y1
lensing analysis
([McClintock et al. 2019, MNRAS 482, 1352](https://ui.adsabs.harvard.edu/abs/2019MNRAS.482.1352M/abstract),
arXiv:[1805.00039](https://arxiv.org/abs/1805.00039)). In this repo it
is implemented at the $\Sigma$/$\Delta\Sigma$ level by
[`src/models/kappa_max.hh`](https://github.com/estevesjh/y3_cluster_cpp/blob/d7feb7504ed5dfcad84f99a1791af8a55c858aa0/src/models/kappa_max.hh)
(`KAPPA_MAX`:
$\kappa = \max\!\big(\Sigma_{\rm NFW},\, b\,\Sigma_{\rm hh}\big)\,
\Sigma_{\rm crit}^{-1}$, from the `haloModel` tables) and
[`src/models/gamma_max.hh`](https://github.com/estevesjh/y3_cluster_cpp/blob/d7feb7504ed5dfcad84f99a1791af8a55c858aa0/src/models/gamma_max.hh)
(`GAMMA_MAX`, the $\Delta\Sigma$ twin). The stacked variant uses the
population-averaged bias
$\langle b\rangle_i = N_i[b]/N_i[1]$ from the `BiasWeightedSel`
diagnostic module ({doc}`../variants`).

**The recipe** (`ct_2hTerm` in `y3_buzzard/haloModel.py`, run per slice
of the 50-point redshift grid):

1. $\xi_{\rm mm}$ from the power spectrum on a fixed grid
   $R \in [10^{-3}, 10^{3}]\ {\rm cMpc}/h$ with **50 log nodes** — the
   correlation must be tabulated well past the BAO scale; 50 nodes is
   the speed/accuracy sweet spot (0.1%).
2. $\xi_{\rm 2halo} = b\,\xi_{\rm mm}$ with $b = 1$: **the tables are
   published unbiased** and the consumer applies $b(M,z)$ or
   $\langle b\rangle_i$.
3. $\Sigma_{2h}$ on the `r_sigma` grid via
   `cluster_toolkit.deltasigma.Sigma_at_R`, which extends the
   $\xi$ table below its inner edge assuming an NFW halo
   ($M_d = 10^{14}\,M_\odot/h$, $c_d = 5$ — a numerical
   parameterisation of the extension, not physics; the inner edge is
   $10^{-3}$ cMpc/$h$, so its imprint is negligible).
4. $\Sigma \to \Delta\Sigma$ needs the full interior mass
   $\bar\Sigma(<R)$. The default method (`dsigma_method='direct'`,
   since the issue #4 fix) evaluates $\Sigma_{2h}$ from the per-$z$
   $\xi$ on a grid extended down to $R = 10^{-3}$ cMpc/$h$ and
   integrates $\bar\Sigma(<R)$ by cumulative trapezoid — the two-halo
   interior mass is *integrated*, not modeled. Validated against
   closed-form NFW/Einasto chains and a converged fiducial anchor
   cross-checked by CLensPy: 0.4% max over
   $R \in [0.5, 20]$ cMpc/$h$, $z \in [0.24, 0.65]$
   (`validations/second_halo_term/`).
5. The historical **NFW-sandwich stabiliser**
   (`dsigma_method='sandwich'`) — *add* an analytic
   $\Sigma_{\rm NFW}$ so `DeltaSigma_at_R`'s interior extrapolation is
   NFW-dominated, then *subtract* the same halo's analytic
   $\Delta\Sigma_{\rm NFW}$ — stays selectable for comparison. With a
   consistent $M_d$ it is dummy-independent (residual $\sim$1%), but it
   cannot recover the two-halo term's own interior mass below the table
   edge (65% max deviation at small $R$ on the same benchmark), which
   is why `direct` is the default. The pre-fix code broke even the
   cancellation (an `Md/10` inconsistency) and then blanked the
   resulting negatives to NaN — 60% of the table; both defects are
   fixed and pinned by `test/halo_model.test.py`.

```{admonition} Composition figure removed — was computed wrong
:class: warning
This page previously embedded a `dsigma_compositions.png` figure
comparing three stacked-lensing compositions. It was pulled after
review: the plotted "1h + ⟨b⟩ΔΣ_2h" curve was a plain **sum**, not the
actual max-model composition this page documents
($\Sigma_{\max} = \max(\Sigma_{\rm NFW},\, b\,\Sigma_{\rm hh})$). The
correct recipe — `Phi_max = max(DSigma_cl, bias * dSigma_hh)`,
population-weighted by $S_{ij}$ — is implemented in
[`shear1h2h_max.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/pipelines/des_y3/src/pipelines/des_y3/shear_1h2h/python/0d/shear1h2h_max.py)
(`compute_shear_max`). Regenerating this figure correctly needs a real
fiducial pipeline dump (`haloModel/{dSigma_hh, bias, dSigma_nfw}` with
`compute_lensing_2h = T`) — not available in this environment (no GPU/
NERSC toolchain, no checked-in dump) — so it isn't fabricated here.
Whoever has pipeline access: run `des_y3.ini` with `compute_lensing_2h
= T`, feed the dump through `compute_shear_max`, and re-embed.
```

The wired production 1h+2h composition modules (`SigmaTotSel` /
`DSigmaTotSel`) are **currently broken** (NaN `dSigma_hh` below
$R \approx 8.6\,h^{-1}$cMpc; interpolation on the wrong radial axis) —
this is why `shear1h2h_max.py` reads `haloModel/dSigma_hh` directly
rather than through those modules. Any 1h+2h comparison must assemble
the term from `BiasWeightedSel` + `xi_nl` directly, or use
`Shear1h2hMax`/`shear1h2h_max.py` as documented here. Details:
{doc}`../modules/historical`.

Full derivation and the composition comparison:
{doc}`../math/index`.

## CosmoSIS setup

Enable inside the `[halo_model]` section (everything else as in the
reference — see {doc}`../cosmology/halo_model`):

```ini
[halo_model]
compute_lensing_1h = T
compute_lensing_2h = T   ; reference run sets F
```

## DataBlock outputs (when enabled)

| DataBlock output | Meaning | Units / shape | Consumed by |
|---|---|---|---|
| `haloModel/Sigma_hh` | two-halo surface density on the `r_sigma` grid, **bias not applied** (consumer multiplies by $b$ or $\langle b\rangle_i$) | `(50, 128)` | 1h+2h variant assembly |
| `haloModel/dSigma_hh` | two-halo excess surface density, bias not applied | `(50, 128)` | same |

`haloModel/Wp_hh` (the chain's internal ξ under a misleading $W_p$ name)
and its mislabeled `haloModel/Rp` axis are **no longer published**
(2026-08: unsupported; their only reader was the legacy
`wp_cluster.cuh` interpolation, which used the wrong radial axis — see
`docs/known_issues/wp_hh_rp_axis_mismatch.md`). The internal ξ stage of
the chain remains pinned by `test/halo_model.test.py`; consumers that
need ξ read `xi_nl`.

