# Traditional 1h+2h shear, max model — GPU (`Shear1h2hMaxGpu.so`)

**Status: validated optimization backend** (2026-08-12). Reference
remains the C++ `Shear1h2hMax.so`; this is its GPU adaptation, ported
faithfully.

Division of labor: HMF/selection-weight construction, the 2-halo
ΔΣ_hh/bias tables, and the centred-NFW table stay a verbatim host-side
port of the C++ class (the same `Interp2D` host models, same
`SelFunction_t`-based z-resolved weight `W2d(bin; lnM, z)`, same
sanitize-NaN-to-0 convention for ΔΣ_hh — see
[docs/dsigma_hh_debug_flag.md](../../../../../../docs/dsigma_hh_debug_flag.md)).
The dominant cost — the miscentred-NFW piece of the 1-halo term
(`y3_cluster::NFW_DSIGMA_MIS`, several transcendentals + a table lookup
per (bin, R, lnM) node, ~11.5k evaluations on the production 12×10×96
grid) — is **one CUDA kernel**, one thread per (bin, R, lnM) node, over
`y3_cuda::NFW_DSIGMA_MIS` (device-resident `Interp2D` table, passed by
value like the other des_y3 GPU modules; Ω_m applied in-kernel). Each
thread reduces its own (bin, R, lnM) contribution over the z nodes and
`atomicAdd`s the partial sum into the (bin, R) accumulator. All (bin, R)
wall results are assembled in `set_sample`; `evaluate()` reads the cache.

## Validation (2026-08-12, shared login-node A100, real pipeline, 12 bins × 10 radii)

Co-run with the C++ `Shear1h2hMax.so` (distinct output sections,
collision-free), identical knobs:

| Channel | max \|GPU/C++ − 1\| |
|---|---|
| vals | 6.4e-15 (floating-point roundoff from parallel summation order) |

**Timing: 8 ms/sample vs 11 ms for the C++ backend — a modest ~1.4×.**
Unlike `ShearPrjFrozenGpu.so`'s ~550k-lookup projection contraction,
this observable's actual compute is small (≈11.5k miscentred-NFW
evaluations + a 737k-element (bin,R,lnM,z) reduction) — kernel-launch
and host-to-device-transfer overhead eat most of the theoretical
parallel win. Inherits the C++ backend's 8.3e-4 accuracy vs the
adaptive reference and its ⚠ caution: ΔΣ_hh itself needs debugging
(see the linked doc); any traditional-shear (1h+2h) result from either
backend is provisional until that's fixed.

Outputs: `shear1h2h_max_gpu/vals` — deliberately NOT the CPU backend's
`shear1h2h_max` section, so the two can co-run for validation.
