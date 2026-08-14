# Number counts — `full_ltmz` reference (Python)

**Status: reference implementation** (validated against the pinned
production pipeline on 2026-08-11). Not a production entry point; the
production counts stage remains `sel_function` + `NumCountsSel.so`
(`fast_mass`).

The explicit triple integral for the expected count in richness bin *i*
and photo-*z* bin *j*:

```text
N_ij = ∫dz ∫dlnM ∫dλ_tr  n(M,z) dV/dΩdz(z) Ω(z)
                          K_j(z) K_i(λ_tr, z) P_HOD(λ_tr | M, z)
```

Unlike the manifest's `NumCountsFullScalarIntegrand` diagnostic, this
reference includes the photo-z kernel K_j, so it computes the maintained
observable. Unlike the production `fast_mass` path, it evaluates every
kernel at the quadrature nodes directly — no intermediate S_ij(lnM, z)
tabulation, no bilinear interpolation.

## Integration variables, limits, quadrature

| Variable | Domain | Quadrature |
|---|---|---|
| z (true) | `[zt_low, zt_high]` (pinned: 0.05–0.80) | fixed GL, `n_z` (default 64) |
| lnM | `[lnm_low, lnm_high]` (pinned: 29.9336–36.7300) | fixed GL, `n_lnm` (default 96) |
| λ_tr | per-(M,z) HOD bracket `[max(0, μ_eff − L·σ_eff), μ_eff + L·σ_eff]`, `L = l_lam` (default 6) | fixed GL, `n_q` (default 32) |

Units follow the pipeline conventions: masses in the HMF_t shifted-axis
convention, volumes in (Mpc/h)^3, Ω(z) in rad^2, counts dimensionless.
Bin ordering is the pinned 12-bin wall (richness fast, z-block slow).

## Composed models

Reused, not copied (approved-plan rule):

- HOD (shifted-Poisson), EMG richness kernel K_i, photo-z kernel K_j, and
  the λ_tr bracket: imported from the maintained
  `src/modules/sel_function/sel_function.py` via `des_y3.shared.sel_kernels`.
- HMF (with `hmf_s`/`hmf_q` nuisance), dV/dΩdz, Ω(z): convention-exact
  replicas of `hmf_t.hh` / `dv_do_dz_t.hh` / `omega_z_des.hh` in
  `des_y3.shared.datablock_models`.

## DataBlock contract

Reads (options): `lam_min, lam_max, zob_min, zob_max, sigma_z` (per-bin
arrays), `zt_low, zt_high, lnm_low, lnm_high`, `n_lnm, n_z, n_q, l_lam`.

Reads (datablock): `cluster_mor/*` (incl. `log10_ratio | log10_M1`),
optional `plob_ltr_params/*`, `mass_function/{m_h,z,dndlnmh}`,
`cluster_abundance/{hmf_s,hmf_q}`, `distances/{z,d_a}`,
`cosmological_parameters/{h0,omega_m,omega_nu,omega_lambda,omega_k}`.

Writes: `numcounts_full_ltmz/vals` — expected counts, shape `(n_bins,)`.
The section name is hardcoded (CosmoSIS `[DEFAULT]` propagation; same
rationale as the fixed-GL C++ evaluators).

## Validation and tolerance

`validate_vs_production.py` replays the real extraction dump
(`docs/figs/real_pipeline_extract.ini`: real HMF, distances, HOD point)
and compares against the production `NumCountsSel.so` values for all 12
pinned bins:

- measured 2026-08-11: max |ratio − 1| = **7.6e-4** (largest in the
  z ∈ [0.50, 0.65] block), pass tolerance 5e-3;
- the residual isolates the production S_ij (192×64) tabulation +
  bilinear interpolation, the only step the two paths do not share;
- the CosmoSIS wrapper was smoke-run inside the extraction pipeline
  (89 ms/sample at default quadrature) with identical results.

Reference cost is not a goal (plan §`full_ltmz`), but the module is cheap
enough (<0.1 s) to co-run with production for cross-checks.

## Other backends

The same integral is implemented as adaptive integrations in
[`../cpp/`](../cpp/README.md) (Cuhre, `NumCountsFullLtmz.so`; agrees with
this reference to 4.9e-4 at `eps_rel = 1e-4`) and
[`../cuda/`](../cuda/README.md) (PAGANI, `NumCountsFullLtmzGpu.so`).
Three backends, three different quadrature strategies over one set of
kernels — their mutual agreement is the strongest internal check the
`full_ltmz` contract has.
