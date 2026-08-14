# Projection shear, frozen-physics fast path — GPU (`ShearPrjFrozenGpu.so`)

**Status: validated optimization backend** (2026-08-12). Production
remains `ShearPrjFrozenPhysics.so`; this is its GPU adaptation at the
plan owner's direction — the mock_mcmc_buzzard.ini algorithm, ported
faithfully.

Division of labor: everything cheap is a verbatim host-side port of
the frozen class (the `sp_detail::build_theta_grid` θ grid, the
ring+wings z grid with the 40-iteration χ inversion, the b_sel
plateaus/sigmoid, the frozen mass shapes with the r_s-anchored a_b(z)
drift, ψ(θ) — all on the same immutable `Interp1D`/`Interp2D` host
models). The dominant cost — the ΔΣ_mis cache and its mass
contraction, ~550k single-kernel NFW table lookups + ~1.1M FMA per
sample — is **one CUDA kernel** over `y3_cuda::NFW_DSIGMA_MIS`
(device-resident `quad::Interp2D` table, passed by value like the
PAGANI integrands; Ω_m applied in-kernel). All 180 wall results are
assembled in `set_sample`; `evaluate()` reads the cache.

## Validation (2026-08-12, login-node A100, real pipeline, 180-point wall)

Co-run with production `shear_prj_frozen_physics` (distinct output
sections, collision-free), identical knobs (`include_omega_z = 0`,
production wall):

| Channel | max \|GPU/production − 1\| |
|---|---|
| rnd | 1.2e-14 |
| cl | 2.2e-11 |
| vals | 1.5e-11 |

**Timing: 8 ms/sample vs 81 ms production — 10×.** The kernel's
few-MB device footprint runs even on a nearly-full shared GPU (where
PAGANI's workspace allocation fails) — the practical difference
between a fixed-grid contraction and an adaptive integrator on device.

Outputs: `dsigma_prj_frozen_gpu/{vals,rnd,cl}`,
`shear_prj_frozen_gpu/{vals,rnd,cl}` — deliberately NOT the production
`shear_prj` alias, so the two modules can co-run for validation. A
production promotion would add that alias (one line) after review.
