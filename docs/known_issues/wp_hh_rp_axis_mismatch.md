# ✅ RESOLVED by removal — `haloModel/Wp_hh` and `haloModel/Rp` are no longer published

**Raised 2026-08-19** while fixing issue #4 (`dsigma_hh_debug_flag.md`);
**resolved 2026-08-19 by dropping both keys** (user decision: `Wp_hh`
is unsupported — nothing in the reference pipeline uses it, and its
only reader was the broken legacy interpolation below).
`halo_model_cosmosis.py` now publishes only `Sigma_hh`/`dSigma_hh`
(on `r_sigma`) under `compute_lensing_2h = T`. The chain's internal ξ
stage remains validated by
`test/halo_model.test.py::test_production_xi_matches_clenspy_per_z`;
consumers that need ξ read `xi_nl`. Legacy `wp_cluster.cuh` /
`sigma_buzzard_y3/avgWpCentBu.cu` will now fail loudly
(BlockNameNotFound) instead of silently interpolating on the wrong
axis — intended, since they are unsupported.

Original write-up below for history.

---

# ⚠ `haloModel/Rp` is not the axis of the 2h tables (`wp_cluster.cuh` interpolates on the wrong grid) [HISTORICAL]

## The mismatch

`y3_buzzard/halo_model_cosmosis.py` publishes, under `compute_lensing_2h = T`:

- `haloModel/Rp` = the **`Radii`** grid (`Radii_min..Radii_max`,
  1.0–35.0 cMpc/h in the fiducial ini, 128 pts) — the xi_hm grid;
- `haloModel/Wp_hh`, `Sigma_hh`, `dSigma_hh` = tables evaluated on the
  **`R_perp`** grid (0.1–20.0 cMpc/h, 128 pts), published separately as
  `haloModel/r_sigma`.

Both grids have 128 points, so nothing throws — the mismatch is silent.

## Affected consumers

- `src/models/wp_cluster.cuh:44-51` builds
  `Interp2D("Rp", "z", "Wp_hh")` — interpolates `Wp_hh` on the wrong
  radial axis (legacy `sigma_buzzard_y3` modules).
- The modern consumers are correct: `Shear1h2hMax*` and
  `src/pipelines/shared/lensing_profiles.py` pair the 2h tables with
  `r_sigma`.
- `docs/source/observables/second_halo_term.md`'s DataBlock table
  previously described `Rp` as "radius grid of the 2h tables" —
  the docs enshrined the wrong-axis assumption.

## Candidate fix (deferred)

One line in `halo_model_cosmosis.py` (`block[section_name, "Rp"] =
R_perp` — or publishing `Wp_hh` on the `Radii` grid it claims) — but
either direction silently changes what legacy readers see, so it needs
an audit of every `Rp`/`Wp_hh` consumer (`wp_cluster.cuh`,
`sigma_buzzard_y3/avgWpCentBu.cu`, any y1-era notebooks) first.
