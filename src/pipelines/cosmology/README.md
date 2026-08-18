# `src/pipelines/cosmology/` — shared halo-model physics

Canonical, survey-agnostic home for the halo-model building blocks that
used to be scattered across `y3_buzzard/`: halo bias, mass-concentration
relations, the analytic NFW profile, the frozen projection-effect
coefficients, the selection-bias closure, and Sigma_crit^-1. Consumers
under `src/pipelines/des_y3/` (and, going forward, DES Y6) import from
here instead of reaching into `y3_buzzard/`.

## Why these are copies, not moves

The `y3_buzzard/` originals are left in place, untouched. This repo has
no visibility into the CosmoSIS `.ini` pipelines in the sibling repos
that may still reference `y3_buzzard/<file>.py` by path (see the
top-level `CLAUDE.md`, "CosmoSIS pipeline runs") — moving or deleting
those files could silently break a production pipeline this tree can't
see. `y3_buzzard/` itself is documented as local/scratch and
non-canonical (`CLAUDE.md`), so this directory is the new canonical
source; `y3_buzzard/` keeps working as-is for whatever already depends
on it. This mirrors the same pattern the rest of `src/pipelines/`
follows for the C++/CUDA side: new code instantiates the existing
immutable models rather than editing them in place.

## Contents

| File | Ported from | Contents |
| --- | --- | --- |
| `halo_model.py` | `y3_buzzard/haloModel.py` | `lensingModel` (1h NFW + 2h via `ct_2hTerm`), `biasModel` (Tinker et al. 2010), `scaleShiftCosmo` |
| `concentration.py` | `y3_buzzard/haloModel.py` + `y3_buzzard/mass_concentration.py` | `child18_mass_concentration` (M200c-native, table-driven per halo sample), `duffy_concentration_relation`, `peakHeight_nonLinearMass`, `c_from_m200` (M200m-native, via the `hydro_mc` M200m->M200c conversion) |
| `nfw_model.py` | `y3_buzzard/nfwModel.py` | Analytical Wright & Brainerd (2000) NFW Sigma/DeltaSigma |
| `hydro_mc.py` | `y3_buzzard/hydro_mc.py` | Vendored Ragagnin+2020 mass-definition/concentration conversion library (external code, verbatim) |
| `prj_params.py` | `y3_buzzard/prj_params.py` | Frozen Costanzi-2026 EMG projection-effect coefficients (`PrjParams`) |
| `bsel.py` | `y3_buzzard/bsel.py` | Canonical exact-wall selection-bias closure. Reads shared `lambda_edges` and `PHOD`, then writes one `b_small/b_large` pair per C++ `(lambda_bin, zo_low, zo_high)` row; the Buzzard path is a compatibility shim. |
| `sigma_crit_inv.py` | `y3_buzzard/buildSigmaCritInv.py` | Sigma_crit^-1(z_lens, R) from the beta lookup table + cosmological shift; only change from the original is dropping an unused `setup_bins.zmeans` import |

`y3_buzzard/massconcen.py` is a byte-for-byte duplicate of
`mass_concentration.py`'s `c_from_m200` (plus a dead,
unconditionally-raising `c_from_m200_ragagnin`) and was not ported.

**Halo mass function (HMF): not yet ported.** Explicitly out of scope
for this pass — the reviewer flagged it TBD; it needs its own
consolidation once the HMF story across `y3_buzzard`/`mf_tinker`/the
Fortran module is settled.

## Import convention

Same convention as `des_y3`/`shared`: put `<repo>/src/pipelines` on
`sys.path`, then import this directory as a top-level `cosmology`
package (`from cosmology.prj_params import PrjParams`, etc.) — not
`from pipelines.cosmology import ...`, since `src/pipelines` itself has
no `__init__.py` and is never imported as a package.
