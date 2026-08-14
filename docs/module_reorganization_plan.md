# Proposal: Layout for New DES Y3 Pipeline Implementations

## Status

**Approved 2026-08-11 (J. Esteves).** The eight decisions listed under
[Approval requested](#approval-requested) are accepted as written, and
Phase 2 additions are authorized to proceed one observable and one
integration strategy at a time under `src/pipelines/des_y3/`. The approval
covers additive work only: no existing entry point moves, no existing
template changes, and each implementation lands as its own focused change
validated per the rules below.

This is a narrow proposal that was put up for discussion and approval. It
defines where new implementations associated with the recently documented
DES Y3 (`des_y3`) pipeline should be developed. `mock_mcmc_buzzard.ini` is
the current configuration used to exercise that pipeline; it is not the
pipeline name.

It is **not** a reorganization of existing code. Every current script,
module, model, test, CMake target, Python entry point, and compiled-library
path stays where it is. Approval would apply only to files created for new
implementations.

## Maintenance boundary

The maintained production contract is the DES Y3 pipeline documented in
`docs/source/running.md` and `docs/source/cosmosis/index.md`. Its current
authoritative configuration is
`des-nersc-cluster-scripts/cosmosis-models/mock_mcmc_buzzard.ini` at commit
`9fd24dd`.

There are three distinct scopes.

### Reference implementations for new work

New numerical work will be validated against these recent pipeline
components, which remain at their current paths:

| Pipeline stage | Current implementation | Maintenance role |
|---|---|---|
| `sel_function` | `src/modules/sel_function/sel_function.py` | Shared richness and photo-$z$ selection tensor |
| `NumCountsSel` | `src/modules/num_counts_sel/NumCounts.cc` | Fast number-count observable |
| `Shear1hMisSel` | `src/modules/num_counts_sel/Shear1hMis.cc` | Fast miscentered one-halo shear observable |
| `b_sel_marg` | `src/modules/b_sel_marg_cpu/BSelMargIntegrand.cc` | Selection-bias marginalization operators |
| `bsel` | `y3_buzzard/bsel.py` | Selection-bias plateau assembly |
| `shear_prj_frozen_physics` | `src/modules/sigma_prj_cpu/ShearPrjFrozenPhysics.cc` | Current projection-shear validation baseline |

The shared C++ models composed by these entry points are maintained as
dependencies of this pipeline. This includes, among others, `MOR_HOD_t`,
`HMF_t`, the richness and redshift kernels, `SelFunction_t`, the fixed-GL
$N$-operators, `P_operator`, the projection evaluators, and the centered and
miscentered NFW models.

Those dependencies remain under `src/models` and `src/utils`. This proposal
does not reorganize or classify the shared model layer.

### Path-stable pipeline support modules

The following repository-owned stages remain part of the maintained pipeline
contract, but are not targets of the new folder layout:

- `cp_camb`;
- `halo_model`;
- `average_sigma_crit_inv`;
- `likelihoods`.

They should receive compatibility fixes required by the DES Y3 pipeline,
but otherwise remain at their current paths and retain their current
interfaces.

The CosmoSIS Standard Library stages `consistency`, `GrowthFactor`, and
`MfTinker` are external dependencies and are outside this source-layout plan.

### Existing code outside the new-work scope

Every other current module, model, CUDA implementation, example, test, and
configuration is out of scope. They remain exactly where they are, with the
same paths, registrations, default build behavior, and interfaces. This plan
makes no maintenance or support classification for them and requires no
changes from their users.

## Forward-looking organization

New implementations belonging to the maintained observable family should be
organized as:

```text
observable -> integration strategy -> language/backend
```

The proposed namespace for new work is:

```text
src/pipelines/des_y3/
├── selection/
│   ├── selection_function/
│   │   └── python/
│   └── selection_bias/
│       ├── marginalization/cpp/
│       └── assembly/python/
│
└── observables/
    ├── number_counts/
    │   ├── full_ltmz/{python,cpp,cuda}/
    │   └── fast_mass/{python,cpp,cuda}/
    ├── shear_1h2h/
    │   ├── full_ltmz/{python,cpp,cuda}/
    │   └── radial_series/{python,cpp,cuda}/
    └── shear_projection/
        ├── full_ltmz/{python,cpp,cuda}/
        └── radial_series/{python,cpp,cuda}/
```

This tree is exclusively for newly created maintained work, not a template to
impose on existing modules. Empty language directories should not be created.
A directory is added only when it contains a runnable implementation or a
substantive design document.

The current production entry points do not move under this proposal. Fixes to
existing implementations continue to be made at their current paths; new
reference or alternative implementations use the namespace above.

## Numerical strategies to maintain

The maintained observable families should use the following strategy names.

### `full_ltmz`

A readable reference calculation with explicit integration over
$\lambda_{\rm true}$, $\ln M$, and $z$. Projection shear may retain an
additional angular coordinate where required by its physical definition.

The full implementation is intended for validation and scientific
cross-checks. It need not meet production timing before it is documented as a
reference.

### `fast_mass`

A number-count calculation in which richness and redshift have been
marginalized or tabulated before the final mass integral. For counts the
operator is $f=1$, so no radial approximation is needed.

Extension (plan owner, 2026-08-12): the same name also covers the
shear observables' exact-redshift-contraction fast path — the $z$
integral (and for the projection observable the per-$(\theta,M)$
clustered weight) done exactly on fixed GL nodes *outside* the radial
operator, with no radial-series truncation and no frozen-physics
approximation. This is what the production `Shear1hMisSel`
(`method = exact`) and the exact `ShearPrjCore` diagnostics compute;
`fast_mass` implementations under the namespace state that algorithm
explicitly per observable.

### `radial_series`

This is the fast strategy for shear observables. It has three distinct
steps:

1. generate the unit-amplitude radial-series functions through $\ell=3$
   offline and store them as a versioned data table;
2. compute every MCMC-dependent redshift weight and its population moments
   before evaluating the radial operator;
3. assemble the observable from the offline radial table, the analytic
   amplitude, and the per-sample population moments.

There is no redshift-freezing approximation in this strategy. Let $q$ denote
any additional non-radial coordinate: it is absent for one-halo shear and may
represent $\theta$ for projection shear. Write the full observable as

$$
O_{ij}(R)=\int dq\int d\ln M\int dz\;
\mathcal S_{ij}(\ln M,z,q)\,\Phi(R,\ln M,q),
$$

where $\mathcal S_{ij}$ contains the HMF, volume, selection, photo-$z$,
geometry, bias, and any other non-profile factors. When the radial profile
$\Phi$ has no remaining true-$z$ dependence, compute the exact redshift
weight first:

$$
W_{ij}(\ln M,q)=\int dz\;\mathcal S_{ij}(\ln M,z,q),
$$

so that

$$
O_{ij}(R)=\int dq\int d\ln M\;
W_{ij}(\ln M,q)\,\Phi(R,\ln M,q).
$$

The derivation, synthetic tests, real-pipeline tests, and timing study are in
the [radial-factorization study](shear1h_radial_factorization.tex). That
document is the design reference for the exact redshift contraction and the
moment expansion used here.

The study evaluates truncations through both the second and third central
moments. In its tested, nearly symmetric mass weights, $\ell=2$ already meets
the stated sub-percent target and the $\ell=3$ term does not improve the
result. This proposal nevertheless supports moments through $\ell=3$ as the
maximum retained order established by the study; it does not require terms
beyond the third moment or reopen the choice among unrelated basis
expansions.

Let $y=\ln r_s(M)$ be the scale coordinate of the radial profile, with the
mass-to-scale-radius Jacobian absorbed through
$W_{ij}(y,q)\,dy=W_{ij}(\ln M,q)\,d\ln M$. For each bin and any fixed $q$,
define

$$
N_{ij}(q)=\int dy\;W_{ij}(y,q),
\qquad
\bar y_{ij}(q)=\frac{1}{N_{ij}(q)}
\int dy\;W_{ij}(y,q)y,
$$

and the normalized central moments

$$
\mu_{ij,\ell}(q)=\frac{1}{N_{ij}(q)}
\int dy\;W_{ij}(y,q)
\bigl[y-\bar y_{ij}(q)\bigr]^\ell,
\qquad 0\leq\ell\leq3.
$$

The radial profile is expanded in $y$ about $\bar y_{ij}$:

$$
\Phi(R,y,q)\simeq
\sum_{\ell=0}^{3}
\frac{[y-\bar y_{ij}(q)]^\ell}{\ell!}
\left.\frac{\partial^\ell\Phi(R,y,q)}{\partial y^\ell}
\right|_{y=\bar y_{ij}(q)}.
$$

After integrating over the precomputed weight,

$$
O_{ij}(R)\simeq\int dq\;N_{ij}(q)
\sum_{\ell=0}^{3}\frac{\mu_{ij,\ell}(q)}{\ell!}
\Phi^{(\ell)}\bigl(R,\bar y_{ij}(q),q\bigr).
$$

Because $\mu_{ij,0}=1$ and $\mu_{ij,1}=0$, the evaluated expression is

$$
O_{ij}(R)\simeq\int dq\;N_{ij}(q)
\left[
\Phi+\frac{\mu_{ij,2}}{2}\Phi^{(2)}
+\frac{\mu_{ij,3}}{6}\Phi^{(3)}
\right]_{y=\bar y_{ij}(q)}.
$$

### Offline unit-profile table

The radial derivatives in the last expression must not be recomputed at every
MCMC sample. This is possible when all sample-dependent amplitude factors are
separable from the scale-radius dependence:

$$
\Phi(R,y,q;\boldsymbol\vartheta)
=A_{\rm sample}(\boldsymbol\vartheta)\,A_0(y)\,
u\bigl(x=R e^{-y},x_q=q e^{-y}\bigr),
$$

where $\boldsymbol\vartheta$ denotes parameters that enter the profile only
through $A_{\rm sample}$. The fixed factor $A_0(y)$ contains any known
scale-radius dependence of the normalization. Other sample parameters may
change the population weight $W_{ij}$ or the query coordinate $q$, but they
must not change the definition of $A_0$ or $u$.

For the current miscentering tables, $u(x,x_{\rm mis})$ is fixed by the table,
concentration, and kernel choice; it does not depend on cosmology or HOD
parameters. A miscentering parameter may change the queried value of
$x_{\rm mis}$, but that does not require regenerating the table. In the
current `NFW_DSIGMA_MIS` convention,

$$
A_{\rm sample}=\rho_{\rm mult},
\qquad
A_0(y)=2e^y\delta_c\rho_{\rm crit}\,10^{-12},
$$

with $e^y=r_s$. The implementation interpolates $u$ first and multiplies by
$A_{\rm sample}A_0$ afterward, so it satisfies the required separation.

An offline generator should differentiate the unit-normalized profile family
once and tabulate

$$
U_\ell(y,x,x_q)=
\frac{1}{\ell!\,A_0(y)}
\frac{\partial^\ell}{\partial y^\ell}
\left[A_0(y)u\bigl(R e^{-y},q e^{-y}\bigr)\right],
\qquad 0\leq\ell\leq3,
$$

using the fixed profile conventions. It then follows that

$$
\frac{1}{\ell!}\Phi^{(\ell)}(R,y,q;\boldsymbol\vartheta)
=A_{\rm sample}(\boldsymbol\vartheta)A_0(y)U_\ell(y,x,x_q).
$$

At runtime the radial series is therefore obtained by interpolating $U_0$,
$U_2$, and $U_3$ and restoring the analytic amplitude. $U_1$ is retained for
validation even though its population coefficient $\mu_{ij,1}$ vanishes by
construction. For the current NFW normalization, $A_0(y)\propto e^y$, so the
explicit $y$ dependence cancels from $U_\ell$ and the derived table needs only
the $(x,x_{\rm mis})$ axes. A future fixed profile with a more complicated
$A_0(y)$ may require an additional $y$ axis while remaining reusable across
MCMC samples.

The derived data should live under `data/radial_series/` and should record:

- the dimensionless axes, such as $\ln x$ and $\ln x_{\rm mis}$, plus a
  $y$ axis only when the fixed normalization requires it;
- $U_\ell$ for $0\leq\ell\leq3$;
- concentration, density convention, miscentering kernel, units, and radial
  domain;
- the generator version and the source-table checksum;
- interpolation and truncation tolerances.

The table is generated once, committed as data, and loaded once at module
construction. CPU and CUDA implementations must read the same derived data.
Changing cosmology or HOD parameters must not regenerate the table.

This reuse rule is conditional on the separation above. If a future profile
has sample-dependent parameters that change its dimensionless shape, or if
its sample and scale-radius dependences cannot be factored, it requires a
separate validated strategy and must not silently reuse this offline table.

The quantities $N_{ij}$, $\bar y_{ij}$, $\mu_{ij,2}$, and $\mu_{ij,3}$ are a
different kind of moment: they describe the MCMC-dependent halo population,
not the unit radial profile. They still belong in the per-sample setup/cache
step because the HMF, selection function, and HOD parameters change. The
evaluator's `operator()` or `evaluate()` function should only interpolate the
offline $U_\ell$ table, restore the amplitude, and assemble the three surviving
series terms for the requested radius.

The term `radial_series` applies to any sufficiently smooth radial profile.
NFW and the miscentered one-halo mixture are the cases validated in the
[radial-factorization study](shear1h_radial_factorization.tex). The same
moment interface should accept $1h+2h$, projection, or other tabulated
profiles, but the existing study does not by itself validate those extensions.
Each new profile must confirm that truncation at or below $\ell=3$ meets its
required tolerance.

## Current-to-future mapping

| Observable | Current maintained path | Future reference | Future fast path |
|---|---|---|---|
| Number counts | `NumCountsSel` | Full $(\lambda_{\rm true},\ln M,z)$ implementation | `fast_mass`: current fixed-GL, redshift-contracted mass calculation |
| Halo shear | `Shear1hMisSel` | Full $(\lambda_{\rm true},\ln M,z)$ $1h+2h$ implementation | `radial_series`: precomputed redshift weights and moments through $\ell=3$ |
| Projection shear | `ShearPrjFrozenPhysics` as a validation baseline | Full redshift-, mass-, and angle-resolved implementation | `radial_series`: exact redshift weights and moments through $\ell=3$; no redshift freeze |

Python, C++, and CUDA directories express possible backends, not promised
implementations. A backend is documented as available only after it is
runnable and validated against the same scientific contract.

## Compatibility rules

The maintenance boundary is deliberately smaller, but the current pipeline
paths are still public interfaces. Work in scope must follow these rules:

1. Keep the current CosmoSIS module labels, DataBlock sections, keys, units,
   shapes, and bin ordering stable during structural work.
2. Keep the current Python entry-point and compiled `.so` paths working for
   `mock_mcmc_buzzard.ini`.
3. Add new implementations without moving, renaming, or wrapping the current
   production entry points.
4. Keep `models/...` and `utils/...` include prefixes stable.
5. Do not combine a structural change with a physics, constructor, or
   DataBlock-contract change.
6. Do not modify unrelated module registrations, examples, or existing tests.

## Existing C++ templates are immutable

The existing C++ module and integration templates are fixed dependencies for
this work and must not be modified. This includes the infrastructure in:

- `src/utils/CosmoSISScalarEvaluatorModule.hh`;
- `src/utils/CosmoSISScalarIntegrationModule.hh`;
- `src/utils/CosmoSISVectorIntegrationModule.hh`;
- `src/utils/OneDIntegrationModule.hh`;
- `src/utils/CosmoSISSICUDAModule.hh`;
- `src/utils/module_macros.hh` and `src/utils/cuda_module_macros.cuh`;
- existing shared operator templates such as `n_operator_sel_t.hh` and
  `n_operator_sel_gl_t.hh`.

A new C++ implementation should satisfy an existing template's interface and
instantiate it from a thin module driver. If an existing template cannot
express the new algorithm, the implementation should add a new scoped
adapter or template under `src/pipelines/des_y3`; it must not generalize or
edit the existing template in place.

Any proposed change to an existing C++ template is outside this plan and
requires a separate design, compatibility review, and approval.

## Validation required for maintained work

Each new implementation must document:

- its integration variables, limits, units, and binning;
- its required DataBlock inputs and produced outputs;
- the HOD, HMF, photo-$z$, selection, miscentering, and survey-area choices it
  composes;
- its numerical tolerance against the corresponding reference;
- its language/backend and production, reference, experimental, or planned
  status.

Accuracy policy (plan owner, 2026-08-12): numerical *accuracy* is
always quoted against the corresponding `full_ltmz` fiducial — the
fully explicit calculation, itself certified by internal quadrature
convergence and by cross-backend agreement — never against a
production implementation, which carries its own approximations
(selection tabulation, frozen physics). Agreement with a production
entry point is reported separately as an algorithm-identity or
compatibility check.

Validation should be proportional to the change and limited to the maintained
pipeline surface:

- targeted unit tests for new or changed shared logic;
- reference-versus-fast numerical comparisons;
- verification that truncation through $\ell=3$ reproduces the full radial
  integral within the tolerance established by the radial-factorization
  study;
- verification that the offline $U_\ell$ tables reproduce direct derivatives
  of the source radial profile over their complete interpolation domain;
- backend equivalence tests when more than one backend exists;
- the unchanged CTest suite on Perlmutter;
- a `mock_mcmc_buzzard.ini` smoke run and data-vector comparison.

Existing tests outside this maintenance scope remain registered and
untouched. Passing the maintained-pipeline checks is not permission to remove
or rewrite them.

## Delivery sequence

### Phase 1: Maintenance manifest — implemented

The Phase 1 snapshot is recorded in the
[DES Y3 maintenance manifest](des_y3_maintenance_manifest.md).

- Record the exact source path, build target, DataBlock contract, and shared
  model dependencies for the six reference stages.
- Pin the authoritative external `.ini` and values configuration.
- Record which current implementation corresponds to each numerical
  strategy.

This phase changes documentation only and is complete.

### Phase 2: Add future implementations

Progress (2026-08-11/12): the namespace exists at `src/pipelines/des_y3/`
with its shared Python model layer, and the first implementations have
landed with their validation records in their own READMEs:

- number counts, `full_ltmz`: Python fixed-GL reference (7.6e-4 vs
  `NumCountsSel.so` across the 12 pinned bins), C++ adaptive-Cuhre
  backend (4.9e-4 vs the Python reference at `eps_rel = 1e-4`), and a
  CUDA/PAGANI backend
  (`observables/number_counts/full_ltmz/{python,cpp,cuda}/`);
- one-halo miscentred shear, `radial_series`: Python and C++ backends
  (`observables/shear_1h2h/radial_series/{python,cpp}/`), with the
  offline $U_\ell$ tables generated and committed under
  `data/radial_series/`, truncation validated to 0.45% ($\ell\le2$) on
  the 12 real bins, and backend equivalence at 1.6e-4.

None of these is a production entry point; the production stages are
untouched.

- Add one observable and one integration strategy at a time under the new
  maintained namespace.
- Start with the full $(\lambda_{\rm true},\ln M,z)$ reference or the
  appropriate `fast_mass`/`radial_series` implementation according to the
  scientific priority.
- Reuse the maintained shared model layer rather than copying HOD, HMF,
  photo-$z$, or lensing constructors into each observable directory.
- Generate and validate the unit-profile $U_\ell$ data once; do not rebuild
  radial basis functions inside an MCMC sample.
- Instantiate or wrap the existing C++ templates without modifying them.
- Validate against the currently documented production output.

Each implementation should have its own focused pull request.

## Approval requested

Approval is requested only for these decisions:

1. use the six recent selection and observable stages in the documented
   `des_y3` pipeline as the validation baseline for new work;
2. keep the four repository-owned support stages working but path-stable;
3. leave every existing script, module, model, test, build target, and
   configuration at its current path and outside this layout decision;
4. use `observable -> integration strategy -> language/backend` only for new
   maintained work;
5. use `full_ltmz`, `fast_mass`, and `radial_series` with the meanings
   defined above;
6. treat the existing C++ templates as immutable dependencies;
7. generate the unit-profile radial-series data once, version it under
   `data/radial_series/`, and never regenerate it inside an MCMC sample;
8. do not relocate existing production entry points as part of this plan.

The Phase 1 documentation manifest is now implemented. Approval of this
proposal would not authorize any Phase 2 source move, deletion, build change,
test change, or physics change; those require separate review.
