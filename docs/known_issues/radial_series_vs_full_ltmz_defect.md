# ✅ RESOLVED — radial series (0d) vs the 3d reference: density-convention mismatch + fixed-c policy decision (issue #5 closed)

**Raised 2026-08-14; root-caused and RESOLVED 2026-08-24.** The original
write-up attributed the offset to the hardcoded `CONC = 4.0` in
`nfw_profile_family.py`. A quantitative decomposition (switching one
ingredient at a time against the dump-verified production profile)
showed that attribution was wrong:

| Effect (per halo, R = 0.2 cMpc/h) | ratio radial_series / 3d |
|---|---|
| concentration only (c=4 vs Child18 at z_halo=0) | 0.91 – 1.00 (small, wrong sign at low M, ~0 at high M) |
| density/boundary convention only (rho_crit/200c vs rho_m0/200m, both c=4) | **1.78 – 1.96** (and the universal ~1.22 large-R floor) |

The dominant root cause: the family's CENTRED profile was pinned to the
*miscentering-table* convention — `r_200` against **rho_crit** (200c)
with amplitude `delta_c * rho_crit` and NO Omega_m (`rho_mult` applied
to the mis component only) — while the 3d reference / production centred
term (`haloModel/dSigma_nfw`) interprets the same mass as **M_200m**:
`r_200` against rho_m0 = Omega_m rho_crit with amplitude
`delta_c * rho_m0`. Two compounding pieces: amplitude x(1/Omega_m)=3.23
and boundary r_s x Omega_m^(1/3)=0.677 with the query x = R/r_s shifted
by 1.478 along the Wright & Brainerd shape. The per-bin population-level
prediction from this decomposition reproduces the observed 1.56–1.86
ratios and the large-R floor; the concentration piece is <= 9%.

## Resolution (owner decision, 2026-08-24): the unified rho_m convention

Every profile component in every backend (CPU, CUDA, Python, radial
series) now uses ONE reference density for BOTH the halo boundary and
the amplitude: `haloModel/rho_m_ref` = Omega_m rho_crit,0
(1 + one_halo_z_density)^3, published by `halo_model_cosmosis.py` — the
identical density `first_halo_term` builds the centred tables with.

- `NFW_DSIGMA_MIS` / `NFW_SIGMA_MIS` (`.hh`/`.cuh`): `set_rho_ref(rho)`
  replaces the removed `rho_mult` machinery; pure normalization factors
  are applied OUTSIDE the profile classes.
- `nfw_profile_family.py` / `shear1h_radial_series` (py+cpp):
  `y_of_lnM`, `A0_of_y`, `r_s_of_lnM` take `rho_ref`; the offline U_ell
  tables are dimensionless in x and unchanged; `u_mix` is a plain f_mis
  blend (no Omega_m on the mis component).
- Physical density rho_m(z) = rho_m0 (1+z)^3 is available in-integrand
  via `[halo_model] one_halo_physical_density` using the exact fixed-c
  identity DSigma_phys(R|z) = (1+z)^2 DSigma_com(R (1+z)) — (1+z)^2 in
  the shear z-weight + the query-radius rescale (exact at live z;
  bin-z_eff in the z-contracted evaluators). Number counts untouched.

## Concentration question: decided (2026-08-27) — closes issue #5

The residual radial-series-vs-3d disagreement after unification was the
genuine remaining content of issue #5: the fixed c=4 family vs Child18
c(M) (<= ~9% at inner radii, vanishing at high mass at z_halo = 0) plus
the ell-truncation/interpolation of the series (~few x 1e-3). The
deliberately-red pin `test_cpp_radial_series_matches_cpp_full_ltmz` was
gated on deciding whether the family should adopt Child18 c(M) too.

Owner decision (2026-08-27): fixed concentration, not Child18 c(M), is
the project's direction for the 1-halo term (see the `concentration_fixed`
knob added to `halo_model.py` / `halo_model_cosmosis.py`, and the Buzzard
mock's own c=5 fixed-concentration switch in `des-nersc-cluster-scripts`).
The family's fixed c=4 is therefore the correct model choice, not a
defect to close by chasing Child18 -- the residual -10.6%..+3.9%
envelope (below) is an accepted, permanent characteristic of the
fixed-c approximation, not an open question. Issue #5 is closed on
this basis.

Follow-up (not done here, tracked separately if it matters later):
`test_cpp_radial_series_matches_cpp_full_ltmz`'s tolerance/red-pin status
should be revisited to reflect "accepted approximation" rather than
"defect pending a decision" -- a wording/tolerance change, not a physics
change.

Measured envelope after the convention migration's pin regeneration
(2026-08-24, 120-point wall on the fixture's pinned z_halo = 0 tables,
regenerated `CPP_RADIAL_SERIES` vs `CPP_FULL_LTMZ`):
**-10.6% .. +3.9%**, sign flipping with radius within each bin (e.g.
bin 0: -10.0% at R = 0.2 rising through zero near R ~ 1.4 to +3.5% at
R = 5 cMpc/h) — the shape signature of a concentration mismatch, no
longer an amplitude offset. Down from the pre-unification 56-86%.
(On z_halo = 0.4 tables the same envelope is -3.6% .. +4.6% — Child18
c at z = 0 sits further from the family's fixed c = 4, so the pinned
legacy-z dump shows the larger residual.)

Historical note: the "~4% shape difference" in the module docstrings and
the earlier "~10% shape residual" were artifacts of shape-only
(R0-normalized) comparisons that hid most of the amplitude convention
mismatch.
