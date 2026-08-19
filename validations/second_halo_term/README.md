# Two-halo term validation harness (issue #4)

Staged validation of the `haloModel/Sigma_hh` / `dSigma_hh` / `Wp_hh`
producer (`y3_buzzard/haloModel.py::ct_2hTerm`), one transform at a time:

    P(k)  -->  xi(r)  -->  Sigma(R)  -->  DeltaSigma(R)

Every stage is validated two independent ways:

1. **Analytic chain bench** — profiles whose full chain is closed-form
   (NFW via Si/Ci + Wright & Brainerd; Einasto n=1/2 = Gaussian; Einasto
   n=1 = exponential; see `common/analytic_profiles.py`, self-checked by
   quadrature in both envs). A failure localizes to one transform.
2. **Fiducial P(k) cross-code comparison** — the same CAMB P(k, z)
   (sigma8-matched to the dump cosmology) pushed through
   cluster_toolkit, CLensPy `TwoHaloTerm`, pyccl (`ccl.correlation`),
   clmm (2h excess surface density), and the production code path.

Run everything: `./run_all.sh` (pre-fix baseline) or
`./run_all.sh after` (post-fix, evaluates `dsigma_method` variants).
Order: 01 → 07; outputs land in `outputs/*.npz` (gitignored), report
inputs in `report/{figs,tables,values.tex}` (script-generated, never
hand-edited).

## Environments

| env | interpreter | provides |
|---|---|---|
| conda `y3cl_je_macos` | `/opt/homebrew/Caskroom/miniforge/base/envs/y3cl_je_macos/bin/python` | camb 1.4.0, cluster_toolkit, production `y3_buzzard` code |
| CLensPy venv | `/Users/jesteves/Documents/Dev/github/CLensPy/.venv/bin/python` | clenspy (editable), mcfit, pyccl, clmm |

P(k) is generated ONCE (01, conda env) and shared as `.npz`; the CLensPy
venv only consumes it — neutralizes the camb 1.4.0 vs 2.0.3 divergence
and the numpy 1.26 vs 2.4 boundary (plain float64 arrays only, loaded
with `allow_pickle=False`).

## Unit / cosmology ledger

Fiducial: Omega_m=0.311049, h=0.6766, sigma8=0.8238, n_s=0.9665,
ombh2=0.022420145751, omch2=0.11997421699944, mnu=0, w=-1 (the dump's
`cosmological_parameters`). RHO_C = 2.77533742639e11 Msun/Mpc^3/h^2.

| code | R / r | M | Sigma, DeltaSigma | rho convention |
|---|---|---|---|---|
| production, cluster_toolkit | cMpc/h | Msun/h | Msun h/pc^2 comoving | rho_m = Omega_m·RHO_C (h-units, comoving, no (1+z)^3) |
| CLensPy `TwoHaloTerm` | Mpc physical-convention (feed k·h, P/h^3, R/h) | — | **dimensionless**; × Omega_m·RHO_C·h^2, /h, /1e12 → Msun h/pc^2 | caller-owned |
| pyccl | Mpc | Msun | — (xi only here) | Omega_c = Omega_m − Omega_b |
| clmm | Mpc physical | Msun | Msun/Mpc^2 physical; comoving bridge ΔΣ_com(R_com) = ΔΣ_phys(R_com/(1+z))/(1+z)^2 | Omega_dm0 = Omega_m − Omega_b |

Every `.npz` carries `units_*` strings; `07_compare.py` checks them
before comparing. CLensPy is **under test** here, not gospel — its
`LensingProfile` has known rho_m/1e12 unit bugs (see the report's
CLensPy issue list); the harness uses `TwoHaloTerm` + its own
conversions only.

## Scripts

| script | env | what |
|---|---|---|
| `common/analytic_profiles.py` | both | closed forms + quadrature self-check (`python analytic_profiles.py`) |
| `01_make_pk_camb.py` | conda | CAMB P(k,z) lin+halofit on the dump grids; sigma8 As-iteration; dump-vs-CAMB consistency |
| `02_chain_bench_ct.py` | conda | chain bench through cluster_toolkit; DS method candidates incl. the Md/10 bug replica |
| `03_chain_bench_clenspy.py` | clenspy | chain bench through mcfit / compute_sigma_grid / cumtrapz / TwoHaloTerm |
| `04_reference_ct.py` | conda | fiducial per-z reference, converged grids + converged DS anchor |
| `05_reference_clenspy.py` | clenspy | fiducial CLensPy / pyccl / clmm references + benchmark pin constants |
| `06_production_eval.py` | conda | drives production `ct_2hTerm` as-is (`--tag before/after`, `--method`) + dump table snapshot |
| `07_compare.py` | conda | gates, DS-method decision, all report figures/tables/values.tex |

`outputs/dump_before/` is a committed verbatim snapshot of the pre-fix
checked-in dump tables so the "before" story stays reproducible after
the code and the dump are regenerated.

## Headline baseline results (pre-fix, 2026-08-19)

- CAMB vs dump emulator P(k): 0.36% max, 0.08% median.
- xi 4-way (ct / CLensPy / pyccl) mutual: ≤1.6% for r ∈ [0.5, 50];
  dump `xi_nl` table is **correct** (0.15% vs per-z linear reference).
- Production reproduces all pinned bugs: Sigma_hh/Wp_hh z-degenerate,
  dSigma_hh 3850/6400 NaN below R = 2.48, Sigma_hh +71..+172% vs true
  z≈0.41 two-halo term.
- DeltaSigma-method decision (vs converged anchor, R ≥ 0.5,
  z ∈ {0.24, 0.41, 0.65}): **direct D = 0.44%** vs sandwich D = 65%
  (the consistent-Md sandwich is dummy-independent — cancellation
  residual 0.95% — but irrecoverably loses the 2h profile's own
  interior mass below the table edge). Winner: **direct**.
