# CosmoSIS Consistency

`Python` · `CosmoSIS Standard Library` · `Cosmology` · module `consistency` · `<1 ms/sample`

Completes the cosmological parameter set: from the sampled subset it derives
every related parameter ($h_0 \leftrightarrow H_0$, $\Omega_c$,
$\Omega_\Lambda$, $\omega_b h^2$, …) and verifies the relations are
consistent. Every downstream module reads the completed
`cosmological_parameters` section it produces.

## Numerical framework

Pure algebraic closure of the Friedmann-parameter relations (e.g.
$\Omega_\Lambda = 1 - \Omega_m - \Omega_k$,
$\omega_i = \Omega_i h^2$); no integration. It fails loudly if the sampled
set over-constrains the relations.

## Script

- Source: [`utility/consistency/consistency_interface.py`](https://github.com/joezuntz/cosmosis-standard-library/blob/main/utility/consistency/consistency_interface.py)
  (CosmoSIS Standard Library; version pinned by your local
  `${COSMOSIS_STANDARD_LIBRARY}` checkout).
- Loaded by CosmoSIS as a Python module (no compiled library).
## CosmoSIS setup

```ini
[consistency]
file = ${COSMOSIS_STANDARD_LIBRARY}/utility/consistency/consistency_interface.py
```

- Requires `COSMOSIS_STANDARD_LIBRARY` in the environment.
- Must be the **first** module in the pipeline: everything else reads the
  completed parameter section.

## Configuration options

None set in the reference pipeline (defaults throughout).

## DataBlock inputs

| DataBlock input | Meaning | Units / shape | Produced by |
|---|---|---|---|
| `cosmological_parameters/*` (sampled subset) | `omega_m`, `h0`, `omega_b`, `sigma8_input`, `n_s`, `mnu`, `w`, `wa`, … as declared in the values file | dimensionless (`mnu` in eV) | sampler (values file) |

## DataBlock outputs

| DataBlock output | Meaning | Units / shape | Consumed by |
|---|---|---|---|
| `cosmological_parameters/*` (completed) | full consistent set: `h0`, `hubble`, `omega_m`, `omega_b`, `omega_c`, `omega_nu`, `omega_k`, `omega_lambda`, `ommh2`, `ombh2`, `omch2`, `baryon_fraction`, … | scalars | every downstream module |

