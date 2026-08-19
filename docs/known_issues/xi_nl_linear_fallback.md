# ⚠ `xi_nl` contains LINEAR ξ in every run — halofit input does not exist yet

**Raised 2026-08-19** during the issue #4 two-halo validation
(`validations/second_halo_term/`), elevated by the user as the priority
item on the shear-projection branch. **Decision: keep linear P(k) for
now; the nonlinear input will be added by updating `cp_camb`** (this
note is the tracking record, not a code change).

## Status of the ξ_NL → shearPrj chain (audited)

- **Producer** — `y3_buzzard/halo_model_cosmosis.py` (≈:183):
  `xi_NL[iz] = ct.xi.xi_mm_at_r(r_xi, k_nl, P_k_nl[iz])` per z slice on
  `r ∈ [1e-3, 1e3]` cMpc/h (128 pts), published as `xi_nl/{r, z, xi_nl}`.
  Numerically **correct**: 0.15% against a converged per-z reference,
  and cluster_toolkit/CLensPy-FFTLog/pyccl agree to ≤1.6% on the same
  P(k).
- **Consumers** — the three `sp_detail` cores in
  `src/models/sigma_prj_t.hh` read `Interp2D("xi_nl","r","z")` and
  evaluate `.clamp(dchi, zob)` with `chi` consistently converted to
  cMpc/h (`* h0_`, :415/:530) — units and grid coverage are **clean**.
- **Input P(k)** — the physics gap. `halo_model_cosmosis.py` (:132-137)
  uses `matter_power_nl` if present, else silently aliases the linear
  P(k). **No nonlinear P(k) exists anywhere**: the camb-emulator models
  directory ships only `camb_linear_*` emulators, and no production ini
  sets `cp_camb`'s (already-implemented) `nonlinear_pk_path` option.
  So the table named `xi_nl` has contained ξ_lin in every run to date.

## Why it matters

The paper (DES Y1 optical-selection draft, §methods) is explicit: the
ξ_NL entering the two-halo lensing terms (I₂ ≡ 𝒫[b ξ_NL],
I₁ ≡ 𝒫[b ξ_NL σ(θ)]) is "obtained from the corresponding **halofit**
power spectrum". Measured impact at the fiducial cosmology, z = 0.41
(CAMB halofit vs linear, `validations/second_halo_term/`):

- ξ(r = 1 cMpc/h): halofit/linear = **3.40**
- ΔΣ_2h(R = 3 cMpc/h): **2.84**

ratio → 1 only for r ≳ 20 cMpc/h. Every Σ_prj/ΔΣ_prj/shear_prj and
ΔΣ_hh result to date underestimates the correlated-structure term at
small/intermediate separations by these factors (modulated by b_sel and
the radial weighting).

## The fix path (agreed)

1. **Now**: keep the linear P(k) — all validation and mock work
   proceeds with ξ_lin, documented as such.
2. **Next**: update `cp_camb` to provide the nonlinear P(k) (halofit
   emulator `.npz` for `nonlinear_pk_path`, trained in the camb-emulator
   repo — the module plumbing already exists and is exercised:
   `cp_camb.py` writes `matter_power_nl` when the path is given, and
   the `halo_model_cosmosis` fallback then picks it up with **zero
   further code changes**; `xi_nl`, `dSigma_hh` and the whole projection
   branch inherit it automatically).
3. Validation gate ready: `validations/second_halo_term/outputs/
   pk_camb.npz` carries the CAMB halofit reference (`p_k_nl`) on the
   production grids; when the emulator lands, compare its
   `matter_power_nl` against it (and rerun the harness `nl` variants).
