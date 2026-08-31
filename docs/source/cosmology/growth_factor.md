# Growth Factor

`C` · `CosmoSIS Standard Library` · `Cosmology` · module `GrowthFactor` · `<1 ms/sample`

Computes the linear growth factor $D(z)$ and growth rate $f(z)$ for a flat
cosmology. `cp_camb` uses $D(z)$ to reconstruct the redshift evolution of
its $z{=}0$-only emulated power spectrum, and `halo_model` uses it to
redshift-scale the Tinker peak height.


## Numerical framework

Integrates the linear-perturbation growth ODE for
$\delta(a' ) = \delta(a)\, D(a')/D(a)$ in flat $w_0 w_a$CDM. Consumers must
renormalise by $D(0)$: CosmoSIS growth is matter-domination-normalised, so a
raw $D(z)$ ratio without the $D(0)$ division inflates the Tinker peak height
by $\sim 1.3\times$ (a historical bug documented in
{doc}`halo_model <halo_model>`).

## Script

- Source: [`structure/growth_factor/`](https://github.com/joezuntz/cosmosis-standard-library/tree/main/structure/growth_factor)
  — `growthfactor.c` + `interface.c` (C, verified; the `.so` is not
  Fortran).
- Compiled library loaded by CosmoSIS:
  `${COSMOSIS_STANDARD_LIBRARY}/structure/growth_factor/interface.so`.

## CosmoSIS setup

```ini
[GrowthFactor]
; Must precede cp_camb: publishes growth_parameters/{z, d_z} used by
; cp_camb and halo_model_cosmosis.py (Tinker bias nu-scaling).
file = ${COSMOSIS_STANDARD_LIBRARY}/structure/growth_factor/interface.so
zmin = 0.0
zmax = 4.05
dz   = 0.01
```

- Requires `COSMOSIS_STANDARD_LIBRARY`; build the Standard Library once
  with its own `make`.
- Ordering: after `consistency`, **before** `cp_camb` and `halo_model`.

## Configuration options

| Option | Meaning | Units | Reference value |
|---|---|---|---|
| `zmin` | first redshift of the output grid | — | 0.0 |
| `zmax` | last redshift of the output grid | — | 4.05 |
| `dz` | grid spacing (406 points at the reference settings) | — | 0.01 |

## DataBlock inputs

| DataBlock input | Meaning | Units / shape | Produced by |
|---|---|---|---|
| `cosmological_parameters/omega_m` | total matter density today | scalar | `consistency` |
| `cosmological_parameters/omega_lambda` | dark-energy density today | scalar | `consistency` |
| `cosmological_parameters/w`, `wa` | dark-energy equation of state $w(a) = w_0 + (1-a)w_a$ | scalars | `consistency` (fixed at $-1, 0$) |

## DataBlock outputs

| DataBlock output | Meaning | Units / shape | Consumed by |
|---|---|---|---|
| `growth_parameters/z` | redshift grid | `(406,)` | `cp_camb`, `halo_model` |
| `growth_parameters/d_z` | linear growth factor $D(z)$ (matter-domination normalisation, $D(0) \approx 0.76$) | `(406,)` | `cp_camb`, `halo_model` |
| `growth_parameters/f_z` | growth rate $f = d\ln D/d\ln a$ | `(406,)` | — (unused here) |
| `growth_parameters/a` | scale factor of the samples | `(406,)` | — |

