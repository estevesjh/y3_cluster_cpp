# Survey area Ω(z)

`C++` (model, not a module) · `y3_cluster_cpp` · `Selection`

The effective survey solid angle $\Omega(z)$, in rad², enters every
*cluster-count* population integral. It is not a pipeline module: it is a
hard-coded C++ model evaluated inside the count-type operators, and it is
**deliberately absent** from the surface-density operators.

## Script

- Source: [`src/models/omega_z_des.hh`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/models/omega_z_des.hh)
  (`y3_cluster::OMEGA_Z_DES`; CUDA twin `omega_z_des.cuh`, SDSS variant
  `omega_z_sdss.hh`).
- The constructor takes the DataBlock but **reads nothing** — no
  DataBlock inputs, no ini options. The function is three hard-coded
  degree-5 polynomial pieces (the internal names retain a legacy `SDSS_`
  prefix):

$$\Omega(z) = \begin{cases}
P_1(z) & z < 0.504 \\
P_2(z - 0.6) & 0.504 \le z < 0.7 \\
P_3(z) & z \ge 0.7
\end{cases} \qquad [\mathrm{rad}^2]$$

## Numerical framework

$\Omega(z)$ multiplies the volume element in

$$N_i[f] = \int d\ln M\, dz\; \Omega(z)\, \frac{dV}{d\Omega\,dz}\,
\frac{dn}{d\ln M}\, S_{ij}\, f,$$

converting a per-steradian density into a survey expectation. Because
every lensing observable in the likelihood is a *ratio* of count-type
integrals (or a surface density), the area's only first-order effect on
the fit is through the number counts themselves.

## Where it is applied — and where it is not

| Operator | Ω(z)? | Why |
|---|---|---|
| {doc}`NumCountsSijGl <../observables/number_counts>` (DES Y1: `NumCountsSel`) | **yes** | cluster count — area sets the expected number |
| {doc}`Shear1hGl <../observables/shear_halo>` (DES Y1: `Shear1hMisSel`) | **yes** | count-weighted numerator $N_i[\gamma]$; the area cancels only after division by $N_i[1]$ |
| {doc}`b_sel_marg <../systematics/bsel>` | no | the $P[X]$ operators enter downstream only in ratios where $\Omega$ (and the Poisson kernel normalisation $B_i$) cancel; the Python reference has no area weight |
| `ShearPrjEvaluator` (`shear_prj`) | no — hard-excluded | surface density: $\Omega$ cancels between numerator and normalisation (explicit comment in `src/models/sigma_prj_t.hh`) |
| {doc}`ShearPrjGl <../observables/shear_projection>` (`ShearPrjCore`) | **no toggle — never applied** | this core has no `include_omega_z` option at all (verified against its constructor); $\Omega(z)$ is simply absent from the computation |
| DES Y1 `shear_prj_frozen_physics` | **ini-gated, off** | its (different) frozen-specific core defaults to including it (matching the `ShearPrjGsl` diagnostic); the DES Y1 ini sets `include_omega_z = 0`. Verified: with it on, fiducial self-closure breaks ($\log L = -151.7$); off, closure holds ($-0.004$) |

```{note}
`RichnessSelection` (the Python reference) now carries a `SurveyArea`
dataclass (unity/constant/polynomial) for the same convention choice.
Revisit the convention jointly on both sides if a mock data vector is
regenerated with a deliberate survey-area model.
```

