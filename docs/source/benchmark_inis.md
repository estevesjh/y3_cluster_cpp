# Benchmark and precision-reference pipelines

`cosmosis-models/des_y3_cpp0d_fast.ini` and
`cosmosis-models/des_y3_cpp3d_slow_reference.ini` are not the production
pipeline ({doc}`running` covers that, `cosmosis-models/des_y3.ini`). They
exist to run *all* the des_y3 backends of one numerical strategy — fixed
Gauss-Legendre (`0d`, fast) or adaptive Cuhre/PAGANI (`3d`, the precision
reference) — in a single pipeline, so the two families can be timed and
cross-checked against each other. This page summarizes what they measure
and what the measurements say; the full per-observable derivations and
tolerances live in the observable pages linked below.

## The two families

| File | Backends | Cost/sample | Role |
|---|---|---|---|
| `des_y3_cpp0d_fast.ini` | `NumCountsSel`, `Shear1hMisSel`, `ShearPrjGl`, `Shear1h2hMax` (all fixed Gauss-Legendre) | ~0.2-2.8 s | MCMC-viable production backends |
| `des_y3_cpp3d_slow_reference.ini` | `NumCounts3d`, `Shear1h3d`, `Shear1h2hMax3d` (adaptive Cuhre), `shear_prj_cuhre` (adaptive inner integral + fixed-GL outer angle) | seconds to minutes per sample | Precision reference the fixed-GL backends are validated against |

Both run at the same fiducial cosmology and HOD
(`mock_mcmc_widePlanck_values_mis.ini`: widePlanck + the Y3 fiducial
miscentering fractions $f_{\rm mis}=0.22$, $\tau_{\rm mis}=0.17$), so
their outputs are directly comparable sample-for-sample.

## Precision and cost results

The complete cross-language (Python/C++/CUDA), cross-strategy
(`0d`/`2d`/`3d`) precision and cost table — including a direct
python-3d vs cpp-3d vs cuda-3d comparison and a `0d`-fast-path vs
`3d`-reference comparison for each observable — lives in
[`src/pipelines/des_y3/README.md`](https://github.com/estevesjh/y3_cluster_cpp/blob/pipelines/des_y3/src/pipelines/des_y3/README.md)
under "Precision and cost overview". In short: every fixed-GL fast
path agrees with the adaptive `3d` reference at the
$10^{-4}$-$10^{-3}$ level, consistent with the
$\mathrm{eps\_rel}=10^{-4}$ tolerance the adaptive backends are run at.

## Robustness across the prior

A single fiducial-point measurement can hide two things: whether cost
is representative of the full prior volume, and whether the pipeline
behaves sensibly away from that one point. `des_y3_cpp0d_fast_apriori.ini`,
its Python and GPU-backend twins, and `des_y3_cpp3d_slow_reference_apriori.ini`
answer this by drawing hundreds to a thousand samples from the full MCMC
prior (not just the fiducial point) and recording cost and success/failure
for every draw. The same "Precision and cost overview" section in the
des_y3 README summarizes the results; the two headline findings:

- About 30% of prior draws are deliberate, cheap rejections — the
  emulated $P(k)$ is only trustworthy inside CAMB's *trained* parameter
  range, which is narrower than the sampler's declared prior box, so
  those points are rejected before any cluster physics runs. This is
  by design, not a pipeline defect.
- The adaptive `3d` backends' cost varies by 40-60$\times$ across the
  prior (well under a second at most points, tens of seconds at some
  cosmology/HOD corners) — the single fiducial-point cost quoted above
  is not representative of the worst case.

## Reducing the wall for an interactive run

The full `Shear1h3d`/`Shear1h2hMax3d` wall (12 richness/redshift bins
$\times$ 10 radii = 120 points) and the full `shear_prj_cuhre` wall (180
points) both take too long for a login-node session at 25-95 s and
tens of seconds per point respectively. `des_y3_cpp3d_slow_reference.ini`
reduces both: the shear walls to 6 points (inner/mid/outer radius at
the two extreme richness/redshift corners) and the projection wall to
3 points (one richness/redshift bin, three radii). Per-point cost and
precision are measured on these reduced walls; the full-wall wall-clock
is the per-point cost times the point count, not separately re-measured
here.
